import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

# ATTENTION
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
    
class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention.
    Reference:
    https://blog.floydhub.com/attention-mechanism/#bahdanau-att-step1 --> Attention Mechanism
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning --> PyTorch Image Captioning
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, attention_dim)
        print('encoder_out: ', encoder_out.shape)
        print('att1 -> (batch_size, attention_dim): ', att1.shape)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        print('decoder_hidden: ', decoder_hidden.shape)
        print('att2 -> (batch_size, attention_dim): ', att2.shape)
        #att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        att = self.full_att(self.tanh(att1 + att2))  # (batch_size, num_pixels)
        print('att -> (batch_size, num_pixels): ', att.shape)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        print('alpha -> (batch_size, num_pixels): ', alpha.shape)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        print('attention_weighted_encoding -> (batch_size, encoder_dim): ', attention_weighted_encoding.shape)

        return attention_weighted_encoding, alpha

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, decoder_size, vocab_size, attention_size=512, encoder_size=512, dropout=0.5):
        '''
        [See the diagram of the decoder in Notebook 1]
        The RNN needs to have 4 basic components :
            1. Word Embedding layer : maps the captions to embedded word vector of embed_size.
            2. LSTM layer : inputs( embedded feature vector from CNN , embedded word vector ).
            3. Hidden layer : Takes LSTM output as input and maps it 
                          to (batch_size, caption length, decoder_size) tensor.
            4. Linear layer : Maps the hidden layer output to the number of words
                          we want as output, vocab_size.
        '''
        
        super(DecoderRNN, self).__init__()
        self.encoder_size = encoder_size
        self.attention_size = attention_size
        self.embed_size = embed_size
        self.decoder_size = decoder_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        '''
        Embedding layer parameters:
            vocab_size : size of the dictionary of embeddings, 
                basically the number of tokens in the vocabulary(word2idx) for that batch of data.
            embed_size : the size of each embedding vector of captions
        '''
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        '''
        BahdanauAttention layer parameters:
        '''
        self.attention_layer = BahdanauAttention(encoder_size, decoder_size, attention_size)  # attention network
        
        '''
        LSTM layer parameters :
            input_size  = embed_size 
            hidden_size = decoder_size     # number of units in hidden layer of LSTM  
            num_layers  = 1               # number of LSTM layers ( = 1, by default )
            batch_first = True            # input , output need to have batch size as 1st dimension
            dropout     = 0               # did not use dropout 
        
        Other parameters were not changed from default values provided in the PyTorch implementation.
        '''
        print('embed_size: ', embed_size)
        print('encoder_size: ', encoder_size)
        print('decoder_size: ', decoder_size)
        self.lstm_layer = nn.LSTM( input_size = embed_size+encoder_size, 
                             hidden_size = decoder_size, 
                             #num_layers = num_layers, 
                             #dropout = dropout, 
                             batch_first=True )

        self.init_h_layer = nn.Linear(encoder_size, decoder_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c_layer = nn.Linear(encoder_size, decoder_size)  # linear layer to find initial cell state of LSTMCell
        self.linear_fc_layer = nn.Linear(decoder_size, vocab_size)  # linear layer to find scores over vocabulary


    def init_decoder_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, encoder_dim)
        :return: hidden state, cell state
        """
        #print('encoder_out: ', encoder_out)
        print('encoder_out.shape: ', encoder_out.shape)
        #mean_encoder_out = encoder_out.mean(dim=0) # TODO: DA SISTEMARE
        #print('mean_encoder_out: ', mean_encoder_out)
        #print('mean_encoder_out.shape: ', mean_encoder_out.shape)
        #h = self.init_h_layer(mean_encoder_out)  # (batch_size, decoder_dim)
        #c = self.init_c_layer(mean_encoder_out)
        h = self.init_h_layer(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c_layer(encoder_out)
        return h, c

    
    def forward(self, encoder_out, captions):
        '''
        Arguments :
        For a forward pass, the instantiation of the RNNDecoder class
        receives as inputs 2 arguments  :
        -> encoder_out : ouput of CNNEncoder having shape (batch_size, embed_size).
        -> captions : a PyTorch tensor corresponding to the last batch of captions 
                      having shape (batch_size, caption_length) .
        NOTE : Input parameters have first dimension as batch_size.
        '''
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        print ('batch_size: ', batch_size)
        print ('encoder_out.shape: ', encoder_out.shape)
        print ('encoder_dim: ', encoder_dim)
        print ('vocab_size: ', vocab_size)
        print ('captions.shape: ', captions.shape)

        # Flatten image
        #encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        #num_pixels = encoder_out.size(1)
        #print('num_pixels: ', num_pixels)
        print('encoder_out -> (batch_size, encoder_dim): ', encoder_out.shape)
        
        # Discard the <end> word to avoid the following error in Notebook 1 : Step 4
        # (outputs.shape[1]==captions.shape[1]) condition won't be satisfied otherwise.
        # AssertionError: The shape of the decoder output is incorrect.
        # captions.shape: torch.Size([10, 16])
        #captions = captions[:, :-1] 
        captions_length = captions.shape[1]
        captions_length_list = [captions_length for i in range(captions.shape[0])]
        caption_lengths, sort_ind = torch.FloatTensor(captions_length_list).sort(dim=0, descending=True)
        caption_lengths = (caption_lengths).tolist()
        print('caption_lengths: ',caption_lengths)
        
        # Pass image captions through the word_embeddings layer.
        # output shape : (batch_size, caption length , embed_size)
        embeddings = self.embedding_layer(captions) # (batch_size, max_caption_length, embed_dim)
        print ('embeddings -> (batch_size, max_caption_length, embed_dim): ', embeddings.shape)

        # Initialize LSTM state
        #h, c = self.init_decoder_state(encoder_out)  # (batch_size, decoder_dim)
        h, c = self.init_decoder_state(encoder_out)  # (batch_size, decoder_dim)
        print('h.shape -> (batch_size, decoder_dim): ',h.shape)
        print('c.shape -> (batch_size, decoder_dim): ',c.shape)

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, int(max(caption_lengths)), vocab_size).to(device)
        alphas = torch.zeros(batch_size, int(max(caption_lengths))).to(device)
        print('predictions.shape: ', predictions.shape)
        print('alphas.shape: ', alphas.shape)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(int(max(caption_lengths))):
            batch_size_t = sum([l > t for l in caption_lengths])
            print('Ciclo For nro: ', t)
            print('batch_size_t: ', batch_size_t)
            attention_weighted_encoding, alpha = self.attention_layer(encoder_out, h)

            #gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            #attention_weighted_encoding = gate * attention_weighted_encoding

            print('attention_weighted_encoding.shape: ', attention_weighted_encoding.shape)
            print('alpha.shape: ', alpha.shape)
            #print('embeddings[:, t, :].shape: ', embeddings[:, t, :].shape)
            print('embeddings.shape: ', embeddings.shape)
            print('h.shape: ', h.shape)
            print('c.shape: ', c.shape)
            #print('embeddings[:, t, :] + attention_weighted_encoding: ', torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1).shape)
            print('embeddings + attention_weighted_encoding: ', torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1).unsqueeze(1).shape)

            # TODO: problema con c, viene creato come torch.Size([10, 512]) e poi diventa tupla dopo questa chiamata 
            x, (h, c) = self.lstm_layer(torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1).unsqueeze(1), (h.unsqueeze(0), c.unsqueeze(0)))  # (batch_size_t, decoder_dim)
            print('x.shape: ', x.shape)
            print('h.shape: ', h.shape)
            print('c.shape: ', c.shape)
            h = h.squeeze()
            c = c.squeeze()
            print('h.shape: ', h.shape)
            print('c.shape: ', c.shape)
            #print('c: ', c)

            preds = self.linear_fc_layer(x)  # (batch_size_t, vocab_size)
            #preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            #predictions[:batch_size_t, t, :] = preds
            #alphas[:batch_size_t, t, :] = alpha
            predictions[:batch_size_t, t, :] = preds.squeeze()
            alphas[:batch_size_t, t] = alpha.squeeze()
            print('predictions.shape: ', predictions.shape)
            print('alphas.shape: ', alphas.shape)
        
        return predictions #, captions, captions_length_list, alphas

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass