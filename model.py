import torch
import torch.nn as nn
import torchvision.models as models
import torchvision


# ORIGINAL
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        resnet = models.resnet101(pretrained=True)
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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        '''
        [See the diagram of the decoder in Notebook 1]
        The RNN needs to have 4 basic components :
            1. Word Embedding layer : maps the captions to embedded word vector of embed_size.
            2. LSTM layer : inputs( embedded feature vector from CNN , embedded word vector ).
            3. Hidden layer : Takes LSTM output as input and maps it 
                          to (batch_size, caption length, hidden_size) tensor.
            4. Linear layer : Maps the hidden layer output to the number of words
                          we want as output, vocab_size.
        
        NOTE : I did not define any init_hidden method based on the discussion 
               in the following thread in student hub.
               Hidden state defaults to zero when nothing is specified, 
               thus not requiring the need to explicitly define init_hidden.
               
        [https://study-hall.udacity.com/rooms/community:nd891:682337-project-461/community:thread-11927138595-435532?contextType=room]
        '''
        
        super(DecoderRNN, self).__init__()
        
        '''
         vocab_size : size of the dictionary of embeddings, 
                      basically the number of tokens in the vocabulary(word2idx) 
                      for that batch of data.
         embed_size : the size of each embedding vector of captions
        '''
        
        self.word_embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        '''
        LSTM layer parameters :
        
        input_size  = embed_size 
        hidden_size = hidden_size     # number of units in hidden layer of LSTM  
        num_layers  = 1               # number of LSTM layers ( = 1, by default )
        batch_first = True            # input , output need to have batch size as 1st dimension
        dropout     = 0               # did not use dropout 
        
        Other parameters were not changed from default values provided in the PyTorch implementation.
        '''
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = 0, 
                             batch_first=True )
        
        self.linear_fc = nn.Linear(hidden_size, vocab_size)

    
    def forward(self, features, captions):
        '''
        Arguments :
        For a forward pass, the instantiation of the RNNDecoder class
        receives as inputs 2 arguments  :
        -> features : ouput of CNNEncoder having shape (batch_size, embed_size).
        -> captions : a PyTorch tensor corresponding to the last batch of captions 
                      having shape (batch_size, caption_length) .
        NOTE : Input parameters have first dimension as batch_size.
        '''
        
        # Discard the <end> word to avoid the following error in Notebook 1 : Step 4
        # (outputs.shape[1]==captions.shape[1]) condition won't be satisfied otherwise.
        # AssertionError: The shape of the decoder output is incorrect.
        print('original captions.shape: ', captions.shape)
        captions = captions[:, :-1] 
        print('-1 captions.shape: ', captions.shape)
        
        # Pass image captions through the word_embeddings layer.
        # output shape : (batch_size, caption length , embed_size)
        captions = self.word_embedding_layer(captions)
        print('embedded captions.shape: ', captions.shape)
        
        # Concatenate the feature vectors for image and captions.
        # Features shape : (batch_size, embed_size)
        # Word embeddings shape : (batch_size, caption length , embed_size)
        # output shape : (batch_size, caption length, embed_size)
        print('original features.shape: ', features.shape)
        print('features.unsqueeze(1).shape: ', features.unsqueeze(1).shape)
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        print('cat inputs.shape: ', inputs.shape)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the hidden state is not used, so the returned value is denoted by _.
        # Input to LSTM : concatenated tensor(features, embeddings) and hidden state
        # output shape : (batch_size, caption length, hidden_size)
        outputs, _ = self.lstm(inputs)
        print('lstm outputs.shape: ', outputs.shape)
        
        # output shape : (batch_size, caption length, vocab_size)
        # NOTE : First dimension of output shape is batch_size.
        outputs = self.linear_fc(outputs)
        print('linear outputs.shape: ', outputs.shape)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass