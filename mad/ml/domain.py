from mad.ml import distances
import pandas as pd
import numpy as np


class layer:
    '''
    A class to continuously split on training sets to produce a test set.
    '''

    def __init__(self, X, y, d):
        '''
        Define essential variables for the class.

        inputs:
            splitter = The splitting object.
            X = The features.
            y = The target variable.
            d = The domain assignment.
        '''

        # Make sure these are passble through methods.
        self.X = X
        self.y = y
        self.d = d

        self.layers = []  # Save the layers of splits.

    def make_layers(self, splitters):
        '''
        Generate the layers from splitters supplied.

        input:
            splitters = The splitter corresponding to each layer being
                        produced.
        '''

        [self.add_layer(i) for i in splitters]

        return self.layers

    def add_layer(self, splitter):
        '''
        Get the domain specific splits.

        inputs:
            splitter = The splitting object.

        outputs:
            splits = The indexes of the training and testing splits.
        '''

        # If previous layer exists, then make sub splits.
        if self.layers:
            layer = []
            for i in self.layers[self.layer_number]:
                i = np.array([i[0]]).T
                splits = list(splitter.split(i))

                for j, k in splits:
                    indx_train = i[j].ravel()
                    indx_test = i[k].ravel()
                    layer.append((indx_train, indx_test))

            self.layer_number += 1  # Now added layer.
            self.layers.append(layer)

        # If no layers exist, then make the top layer.
        else:
            splits = list(splitter.split(self.X, self.y, self.d))
            self.layer_number = 0
            self.layers.append(splits)

        return self.layers


class builder:
    '''
    Class to use the ingredients of splits to build a model and assessment.
    '''

    def __init__(self, pipe, X, y, d, splitters):
        '''
        inputs:
            pipe = The machine learning pipeline.
            X = The features.
            y = The target variable.
            d = The domain for each case.
            splitters = The splitting oject to create 3 layers.
        '''

        self.pipe = pipe
        self.X = X
        self.y = y
        self.d = d
        self.top_splitter, self.mid_splitter = splitters

    def assess_model(self):
        '''
        Asses the model through nested CV.
        '''

        # Renaming conviance
        X, y, d = (self.X, self.y, self.d)
        top = self.top_splitter
        mid = self.mid_splitter

        o = np.array(range(X.shape[0]))  # Tracking cases

        # In domain (ID) and other domain (OD) splits.
        for id_index, od_index in top.split(X, y, d):

            X_id, X_od = X[id_index], X[od_index]
            y_id, y_od = y[id_index], y[od_index]
            d_id, d_od = d[id_index], d[od_index]
            o_id, o_od = o[id_index], o[od_index]

            # Training and testing splits.
            for tr_index, te_index in mid.split(X_id, y_id, d_id):

                X_train, X_test = X[tr_index], X[te_index]
                y_train, y_test = y[tr_index], y[te_index]
                d_train, d_test = d[tr_index], d[te_index]
                o_train, o_test = o[tr_index], o[te_index]

                self.pipe.fit(X_train, y_train)
                
                # Make predictions for in domain test and other domain test.
                y_id_test_pred = self.pipe.predict(X_test)
                y_od_test_pred = self.pipe.predict(X_od)
                
                # Calculate distances.
                df_id = distances.distance(X_train, X_test)
                df_od = distances.distance(X_train, X_od)

                # Assign boolean for in domain.
                df_id['in_domain'] = [True]*X_test.shape[0]
                df_od['in_domain'] = [False]*X_od.shape[0]

                # Grab indexes of tests.
                df_id['index'] = o_test
                df_od['index'] = o_od

                # Grab the domain of tests.
                df_id['domain'] = d_test
                df_od['domain'] = d_od

                # Grab the true target variables of test.
                df_id['y'] = y_test
                df_od['y'] = y_od

                # Grab the predictions of tests.
                df_id['y_pred'] = y_id_test_pred
                df_od['y_pred'] = y_od_test_pred

                print(pd.DataFrame(df_id))
                print(pd.DataFrame(df_od))
