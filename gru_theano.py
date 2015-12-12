import numpy as np
import theano as theano
import theano.tensor as T


class GRUTheano:
    def __init__(self, word_dim, hidden_dim, bptt_truncate):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((3,hidden_dim))
        c = np.zeros(word_dim)

        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

        #SGD / rmsprop: initialization parameters
        self.mU = theano.shared(name='mU', value=U.astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=W.astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=V.astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=b.astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=c.astype(theano.config.floatX))

        # store theano graph here
        self.theano = {}
        self.__theano_build__()


    def __theano_build__(self):
        U, W, V, b, c = self.U, self.W, self.V, self.b, self.c

        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step(x_t, s_t_prev, U, W, V, b, c):
            z_t = T.nnet.hard_sigmoid(U[0,:,x_t] + W[0].dot(s_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1,:,x_t] + W[1].dot(s_t_prev) + b[1])
            c_t = T.tanh(U[2,:,x_t] + W[2].dot(s_t_prev * r_t) + b[2])
            s_t =  (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev
            o_t = T.nnet.softmax(V.dot(s_t) + c)[0]

            return [o_t, s_t]

        [o, s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info = [None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences = [U,W,V,b,c],
            truncate_gradient = self.bptt_truncate,
            strict = True
        )


        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o,y))
        cost = o_error # total cost, can add regularization here

        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        dV = T.grad(cost, V)
        db = T.grad(cost, b)
        dc = T.grad(cost, c)

        self.predict = theano.function([x],o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x,y], cost)
        self.bttt = theano.function([x,y], [dU,dW,dV,db,dc])

        #SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        #rmsprop cache updates
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.sgd_step = theano.function(
            [x,y,learning_rate,theano.Param(decay,default=0.9)],
            [],
            updates=[
                (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                (self.mU, mU),
                (self.mW, mW),
                (self.mV, mV),
                (self.mb, mb),
                (self.mc, mc)
            ]
        )

    def calculate_total_loss(self, X, Y):
        return np.sum(self.ce_error(x,y) for x,y in zip(X,Y))

    def calculate_loss(self,X,Y):
        #divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y) / float(num_words)


if __name__ == '__main__':
    #just for test
    model = GRUTheano(word_dim=100, hidden_dim=50, bptt_truncate=-1)