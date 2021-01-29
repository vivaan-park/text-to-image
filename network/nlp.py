# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.layers import (Layer, LSTMCell, GRUCell, RNN,
                                     Bidirectional)

class VariousRNN(Layer):
    def __init__(self, n_hidden=128, n_layer=1, dropout_rate=0.5,
                 bidirectional=True, return_state=True,
                 rnn_type='lstm', name='VariousRNN'):
        super(VariousRNN, self).__init__(name=name)
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.return_state = return_state
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.cell_type = LSTMCell
        elif self.rnn_type == 'gru':
            self.cell_type = GRUCell
        else:
            raise NotImplementedError

        self.rnn = RNN(
            [self.cell_type(units=n_hidden, dropout=self.dropout_rate)
             for _ in range(self.n_layer)],
            return_sequences=True, return_state=self.return_state
        )
        if self.bidirectional:
            self.rnn = Bidirectional(self.rnn)