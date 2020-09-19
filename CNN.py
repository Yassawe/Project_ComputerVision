from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential

class model:
    def __init__(self, lr,  convlayers, fc_layers, kernel, m_f, m_s, activation_hidden, activation_last):
        self.model=self.create_model(lr, convlayers, fc_layers, kernel, m_f, m_s, activation_hidden, activation_last)

    def create_model(self, lr, convlayers, fc_layers, kernel, m_f, m_s, activation_hidden, activation_last):
        model = Sequential()
        for i in range(len(convlayers)):
            if i == 0:
                model.add(Conv2D(convlayers[i], kernel[i], activation=activation_hidden, input_shape=(200, 200, 3)))
            else:
                model.add(Conv2D(convlayers[i], kernel[i], activation=activation_hidden))
            model.add(MaxPooling2D(m_f,m_s))

        model.add(Flatten())

        for i in range(len(fc_layers)):
            if i != len(fc_layers)-1:
                model.add(Dense(fc_layers[i], activation=activation_hidden))
            else:
                model.add(Dense(fc_layers[i], activation=activation_last))

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics='accuracy')

        return model

    def train_model(model, args):
        #TODO: определиться как мы будем оформлять трейн датасет
        pass

    def predict(self, img):
        prediction=self.model.predict(img)
        #assumption that class 1 is stone, class 0 is everything else
        if prediction>0.5:
            return True
        else:
            return False

    def dump_weights(self, filename):
        self.model.save_weights(filename)

    def upload_weights(self, filename):
        self.model.load_weights(filename)
