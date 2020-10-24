# ORIGINAL
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, decoder_size, vocab_size, encoder_size=512, dropout=0.25, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.decoder_size = decoder_size
        self.vocab_size = vocab_size
        self.encoder_size = encoder_size
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.word_embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm_layer = nn.LSTM( input_size = embed_size,
                             hidden_size = decoder_size, # number of units in hidden layer of LSTM  
                             num_layers = num_layers,    # number of LSTM layers ( = 1, by default )
                             #dropout = dropout,          # setting dropout value, avoiding overfitting 
                             batch_first=True            # input , output need to have batch size as 1st dimension
                            )

        self.linear_fc_layer = nn.Linear(decoder_size, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear_fc_layer.weight)
        torch.nn.init.xavier_uniform_(self.word_embedding_layer.weight)
    
    def forward(self, features, captions):
        # preparing input caption by delete the <end> character
        captions = captions[:, :-1] 
        
        # passing the feature throught the enbedding layer
        captions = self.word_embedding_layer(captions) # --> output: (batch_size, caption length , embed_size)
        
        # preparing input to the LSTM and calling the LSTM layer
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1) # --> output: (batch_size, caption length, embed_size)
        outputs, _ = self.lstm_layer(inputs) # --> output: (batch_size, caption length, hidden_size)
        
        # passing the output of the LSTM to the linear layer per ottenere le scores
        outputs = self.linear_fc_layer(outputs) # --> output: (batch_size, caption length, vocab_size)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        
        word_out = None
        seq = []
        states_out = states

        for i in range(max_len):

            if word_out == 1:
                break
            
            lstm_layer_out, states_out = self.lstm_layer(inputs, states_out)
            linear_layer_out = self.linear_fc_layer(lstm_layer_out)

            linear_layer_out = linear_layer_out.squeeze(1) 
            single_word_out  = linear_layer_out.argmax(dim=1)     # extract the word with the highest probability
            word_out = single_word_out.item()
            seq.append(word_out)
    
            # passing throught the embedding for the next cycle
            inputs = self.word_embedding_layer(single_word_out).unsqueeze(1)
        
        return seq