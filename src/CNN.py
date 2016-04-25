import os
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import cv2
from osgeo import gdal
from osgeo.gdalconst import *
from skimage.util.shape import view_as_windows
import time

NKERNS = (20, 50)
N_EPOCHS = 1
LEARNING_RATE = 0.1
VECTOR_LAYER = 'tree-poly'

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        # pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                               filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class CNN_ANN(object):
    "Class for handling CNN and ANN"

    def shared_dataset(self, data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    def build_model(self, learning_rate=None, n_epochs=None,
                        nkerns=(20, 50)):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """

        build_model_start_time = time.time()
        self.nkerns = nkerns
        rng = np.random.RandomState(23455)

        # The size of a sliding window that we use for convolution within each block
        self.win_width = 28
        self.win_height = 28
        self.depth = 3

        #block_width is the width of the section of the input image
        #blocks are not slid over the image the same way a window it, they only overlap by win_width-1
        self.block_width = 64
        self.block_height = 64
        self.block_overlap = self.win_width - 1

        # initialization of the training feature and training result
        # X is an array of windows themselves (each is winwidth x winheight x 3 pixels) and there is one per block
        X = np.zeros((self.block_width * self.block_height, 3 * self.win_width * self.win_height))

        # Y is an array of the labels assigned to each window in the block
        Y = np.zeros((self.block_width * self.block_height,), dtype=np.int32)

        print Y.shape
        print X.shape

        # Are these copies or aliases???
        self.training_set_x, self.training_set_y = self.shared_dataset((X, Y))
        self.test_set_x, self.test_set_y = self.shared_dataset((X, Y))

        # compute number of minibatches for training
        self.n_training_batches = self.training_set_x.get_value(borrow=True).shape[0]
        self.n_testing_batches = self.test_set_x.get_value(borrow=True).shape[0]


        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # Reshape matrix of rasterized images of shape (batch_size,win_width*win_height)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input = x.reshape((self.n_training_batches, self.depth, self.win_width, self.win_height))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (win_width-5+1,win_height-5+1)
        # maxpooling reduces this further to (win_width-5+1,win_height-5+1)/2
        # 4D output tensor is thus of shape (batch_size,nkerns[0],(win_width-5+1,win_height-5+1)/2)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(self.n_training_batches, self.depth, self.win_width, self.win_height),
                                    filter_shape=(nkerns[0], 3, 5, 5), poolsize=(2, 2))

        # Construct the second convolutional pooling layer
        # the input image size after the first convolutional pooling layer is x,y = (win_width-5+1,win_height-5+1)/2
        # filtering reduces the image size to (x-5+1,y-5+1)
        # maxpooling reduces this further to (x-5+1,y-5+1)/2
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],(x-5+1,y-5+1)/2)
        self.image_size = (self.win_width - 4) / 2
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                    image_shape=(self.n_training_batches, nkerns[0], self.image_size, self.image_size),
                                    filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

        print "nkerns", nkerns
        # construct a fully-connected sigmoidal layer
        print self.image_size
        self.image_size = (self.image_size - 4) / 2

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function([self.index], layer1.output.flatten(2),
                                     givens={
                                         x: self.test_set_x[self.index * self.n_training_batches: (self.index + 1) * self.n_training_batches],
                                         y: self.test_set_y[self.index * self.n_training_batches: (self.index + 1) * self.n_training_batches]},
                                     on_unused_input='ignore')

    def train_model(self, rgb=None, training_labels=None):
        ###############
        # TRAIN MODEL #
        ###############
        train_model_start_time = time.time()
        print '... training'

        # here we use ANN_MLP in opencv to train the output of the second convolution pooling layer
        ninputs = self.nkerns[1] * self.image_size * self.image_size
        nhidden = 50
        noutput = 2
        self.layers = np.array([ninputs, nhidden, noutput])
        self.ann_model = cv2.ANN_MLP(self.layers, activateFunc=cv2.ANN_MLP_SIGMOID_SYM)

        start = True

        gdal.AllRegister()
        inputs = gdal.Open(rgb, gdal.GA_ReadOnly)
        inputs_data = inputs.ReadAsArray()
        inputs_data = inputs_data.transpose((1,2,0))
        self.width, self.height = inputs_data.shape[:2]

        self.width -= self.width % self.block_width
        self.height -= self.height % self.block_height

        print "Training model image width is", self.width, "height is ", self.height

        print "Processing Tiles..."
        X, Y = np.mgrid[0:self.width:self.block_width, 0:self.height:self.block_height]

        self.tiles = zip(X.ravel(), Y.ravel())
        print "tiles =", self.tiles

        print "training...."

        train_number = 6
        iterations = 3
        # we choose the first train_number tiles as the training data
        train_num = 5

        training_dataset = gdal.Open(training_labels, gdal.GA_ReadOnly)

        for j in range(iterations):
            print "Iteration: ", j
            for i in range(train_number):
                print "train_number =", i
                # if(i == 2):
                #     train_num = 60
                tile_x, tile_y = self.tiles[i + train_num]
                tile_x, tile_y = int(tile_x), int(tile_y)

                xs = min(self.block_width + self.block_overlap, self.width - tile_x)
                ys = min(self.block_height + self.block_overlap, self.height - tile_y)

                print "tile_x:", tile_x
                print "tile_y:", tile_y
                print "xs:", xs
                print "ys:", ys

                data = inputs.ReadAsArray(xoff=tile_x,
                                          yoff=tile_y,
                                          xsize=xs,
                                          ysize=ys)

                data = data.transpose((1, 2, 0)).astype(np.float32)

                prob_data = training_dataset.ReadAsArray(xoff=tile_x,
                                              yoff=tile_y,
                                              xsize=xs,
                                              ysize=ys)

                if len(prob_data.shape) == 3:
                    prob_data = prob_data.transpose((1, 0, 2)).astype(np.float32)

                print "prob_data", prob_data

                for p in range(xs):
                    for q in range(ys):
                        if prob_data[p][q] >= 1:
                            prob_data[p][q] = 255
                        else:
                            prob_data[p][q] = 0

                # Find trees....
                print "Extracting Treess...."
                window_shape = (self.win_width, self.win_height)
                B = view_as_windows(data, (self.win_width, self.win_height, self.depth))
                r, c, s, w, h, d = B.shape
                print B.shape

                prob_data = prob_data[self.win_width / 2:self.win_width / 2 + r, self.win_height / 2:self.win_height / 2 + c]

                X = np.reshape(B, (r * c, 3 * window_shape[0] * window_shape[1]))
                Y = prob_data.flatten()
                Y = np.reshape(Y, (r * c)).astype(np.int32)

                print "X.shape = ", X.shape
                print "Y.shape = ", Y.shape

                self.test_set_x.set_value(np.asarray(X, dtype=theano.config.floatX), borrow=True)
                self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
                self.n_test_batches /= self.n_training_batches
                print "no test batches = ", self.n_test_batches
                print "Training Tile = ", i

                for tm in xrange(self.n_test_batches):
                #for tm in xrange(2):
                    print "n_test_batches =  ", self.n_test_batches
                    sample = self.test_model(tm)
                    condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS
                    nsteps = 200
                    max_err = 0.001
                    criteria = (condition, nsteps, max_err)
                    para = dict(term_crit=criteria,
                                train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                                bp_dw_scale=0.01,
                                bp_moment_scale=0.1)

                    label = Y[tm * self.n_training_batches: (tm + 1) * self.n_training_batches]

                    labels = np.zeros((sample.shape[0], noutput))
                    labels[:, 0] = np.reshape(label.max() - label, (sample.shape[0], 1)).T
                    labels[:, 1] = np.reshape(label, (sample.shape[0], 1)).T
                    labels /= labels.max()
                    print "sample shape =", sample.shape
                    print "labels shape =", labels.shape
                    if start is True:
                        print "sample.shape, labels.shape = ", sample.shape, labels.shape
                        self.ann_model.train(sample, labels, None, params=para)
                    else:
                        self.ann_model.train(sample, labels, None, params=para, flags=cv2.ANN_MLP_UPDATE_WEIGHTS)

                    start = False

            print "Training program took", time.time() - train_model_start_time, "to run"

    def validate_model(self, rgb=None, training_labels=None):
        print "begin validation..."
        error = 0
        train_number = 4
        for i in range(train_number):

            tile_x, tile_y = self.tiles[i]
            tile_x, tile_y = int(tile_x), int(tile_y)

            xs = min(self.block_width + self.block_overlap, self.width - tile_x)
            ys = min(self.block_height + self.block_overlap, self.height - tile_y)

            inputs = gdal.Open(rgb, gdal.GA_ReadOnly)
            data = inputs.ReadAsArray(xoff=tile_x,
                                      yoff=tile_y,
                                      xsize=xs,
                                      ysize=ys)

            data = data.transpose((1, 2, 0)).astype(np.float32)
            self.n_training_batches = self.training_set_x.get_value(borrow=True).shape[0]

            training_dataset = gdal.Open(training_labels, gdal.GA_ReadOnly)
            prob_data = training_dataset.ReadAsArray(xoff=tile_x,
                                          yoff=tile_y,
                                          xsize=xs,
                                          ysize=ys)

            if len(prob_data.shape) == 3:
                prob_data = prob_data.transpose((1, 2, 0)).astype(np.float32)

            xs = min(self.block_width, self.width - tile_x)
            ys = min(self.block_height, self.height - tile_y)

            window_shape = (self.win_width, self.win_height)
            B = view_as_windows(data, (self.win_width, self.win_height, self.depth))
            r, c, s, w, h, d = B.shape
            print B.shape

            prob_data = prob_data[self.win_width / 2:self.win_width / 2 + r, self.win_height / 2:self.win_height / 2 + c]

            X = np.reshape(B, (r * c, 3 * window_shape[0] * window_shape[1]))
            Y = prob_data.flatten()
            Y = np.reshape(Y, (r * c)).astype(np.int32)

            new_test_set_x = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)

            n_test_batches = new_test_set_x.get_value(borrow=True).shape[0]
            n_test_batches /= self.n_training_batches
            result = []

            # Find trees....
            print "Extracting Trees...."

            for ntb in xrange(n_test_batches):
                sample = self.test_model(ntb)
                predictions = np.empty((sample.shape[0], 2))
                print "Sample shape =", sample.shape[0]
                g, prediction = self.ann_model.predict(sample, predictions)
                pred_labels = np.argmax(prediction, axis=1)
                result.append(pred_labels)

            pred = np.reshape(result, (r, c))
            error += np.mean(pred.astype(np.uint8) ^ prob_data.astype(np.uint8))
            print "tile ", i, "error ", np.mean(pred.astype(np.uint8) ^ prob_data.astype(np.uint8))

        print "validation error", error / train_number

    def extract_tree_model(self, rgb=None, output_labels=None):
        all_tiles_time = time.time()

        X, Y = np.mgrid[0:(self.width-self.block_width):self.block_width, 0:(self.height-self.block_height):self.block_height]
        tile_data = zip(X.ravel(),Y.ravel())

        inputs = gdal.Open(rgb, gdal.GA_ReadOnly)

        print "Opening output files for writing...."
        driver = gdal.GetDriverByName('GTiff')
        lblDS = driver.Create(output_labels, self.width, self.height, 1, gdal.GDT_Byte)

        if lblDS is None:
            print "Unable to create labels image"
            return

        lblDS.SetProjection(inputs.GetProjection())
        lblDS.SetGeoTransform(inputs.GetGeoTransform())

        lblBand = lblDS.GetRasterBand(1)

        def on_tile_finished(tile_data, i, result):
            trees = result
            tile_x, tile_y = tile_data[i]
            tile_x, tile_y = int(tile_x), int(tile_y)

            lblBand.WriteArray(trees, xoff=tile_x, yoff=tile_y)

        for i in range(len(tile_data)):
            print "tile num =", i
            tile_time = time.time()

            tile_x, tile_y = tile_data[i]
            tile_x, tile_y = int(tile_x), int(tile_y)

            print "tile_x, tile_y = ", tile_x, tile_y
            xs = min(self.block_width + self.block_overlap, self.width - tile_x)
            ys = min(self.block_height + self.block_overlap, self.height - tile_y)
            print "xs, ys = ", xs, ys

            data = inputs.ReadAsArray(xoff=tile_x,
                                      yoff=tile_y,
                                      xsize=xs,
                                      ysize=ys)

            data = data.transpose((1, 2, 0)).astype(np.float32)

            window_shape = (self.win_width, self.win_height)
            B = view_as_windows(data, (self.win_width, self.win_height, self.depth))
            r, c, s, w, h, d = B.shape

            X = np.reshape(B, (r * c, 3 * window_shape[0] * window_shape[1]))

            new_test_set_x = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)

            n_test_batches = new_test_set_x.get_value(borrow=True).shape[0]
            n_test_batches /= self.n_training_batches
            self.test_set_x.set_value(np.asarray(X,dtype=theano.config.floatX),borrow=True)
            result = []
            # Find trees....
            for j in xrange(self.n_test_batches):
                sample = self.test_model(j)
                predictions = np.empty((sample.shape[0], 2))
                g, prediction = self.ann_model.predict(sample, predictions)
                pred_labels = np.argmax(prediction, axis=1)
                result.append(pred_labels)

            print "result array =", result
            print "r, c shape", r, c
            pred = np.reshape(result, (r, c))
            on_tile_finished(tile_data, i, pred)

            print "Tile prediction took", time.time() - tile_time, "to run"

        print "All tiles processing took", time.time() - all_tiles_time, "to run"

        lblBand.FlushCache()
        lblBand = None
        lblDS = None
        del lblDS

    def process_output(self, raster_data=None, output_label=None, vector_data=None):

        from osgeo import gdal, ogr, osr
        vector_ds = ogr.Open(vector_data)

        tree_polys = vector_ds.GetLayerByName(VECTOR_LAYER)
        raster_ds = gdal.Open(raster_data)
        results_ds = gdal.Open(output_label)
        data = results_ds.ReadAsArray(xoff=0,
                            yoff=0,
                            xsize=self.width,
                            ysize=self.height)

        raster_srs = osr.SpatialReference(raster_ds.GetProjection())
        rgb = raster_ds.ReadAsArray().transpose(1,2,0)  # From color, x, y to  x, y, color (planar to interleaved format)

        import skimage.draw
        ogr.UseExceptions()
        trees_raster = np.zeros_like(rgb[:,:,0])
        tfm = raster_ds.GetGeoTransform()
        _, tfm = gdal.InvGeoTransform(tfm)
        tree_polys.ResetReading()
        for tree in tree_polys:
            geom = tree.GetGeometryRef().Clone()
            geom.TransformTo(raster_srs)
            poly = geom.GetGeometryRef(0)
            points = poly.GetPoints()
            points = [gdal.ApplyGeoTransform(tfm, x, y) for (x,y) in points]
            points = np.asarray(points)
            pixels = skimage.draw.polygon(points[:,1], points[:,0], trees_raster.shape)
            trees_raster[pixels] = 1

        shadows = rgb[:,:,0] < 100

        from skimage.morphology import binary_dilation, disk
        not_trees  = ~binary_dilation(trees_raster, disk(10))

        results = np.zeros_like(trees_raster)
        #print "results = ",results
        #print "data = ",data
        print data.shape
        xs, ys = data.shape
        for p in range(xs):
            for q in range(ys):
                results[p][q] = data[p][q]

        vis = rgb.copy()
        alpha = 0.7
        TP = results & trees_raster != 0
        FP = results & not_trees != 0
        FN = ~results & trees_raster != 0
        TN = ~results & not_trees != 0
        vis[FP, 0] = (1-alpha)*vis[FP, 0]  + alpha*255
        vis[TP, 1] = (1-alpha)*vis[TP, 1]  + alpha*255
        vis[FN, 2] = (1-alpha)*vis[FN, 2]  + alpha*255
        tp, tn, fp, fn = TP.sum(), TN.sum(), FP.sum(), FN.sum()
        total = tp+tn +fp+fn
        print "TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn
        recall =  (tp )/float(tp + fn)
        precision =  tp /float(tp + fp)
        f1 = (2*precision*recall)/(precision+recall)
        print "Recall:", recall*100, "Precision:", precision*100, "F_1-measure:",  f1*100

        #figsize(15, 15)
        #axis('off')
        #imshow(vis)
        import matplotlib.image as mpimg
        mpimg.imsave('Results.tif', vis)
        #title("Results")
        #red_label = mpl.patches.Patch(color='red', label='False positives({:.2f}%)'.format(fp*100./total))
        #green_label = mpl.patches.Patch(color='green', label='True positives({:.2f}%)'.format(tp*100./total))
        #blue_label = mpl.patches.Patch(color='blue', label='False negatives({:.2f}%)'.format(fn*100./total))
        #plt.legend(loc='lower right',fancybox=True, ncol=3,
        #           handles=[red_label, green_label, blue_label])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_site", help="Folder that has input files", type=str)
    p.add_argument("--output", "-o", help="Folder that holds the output files", type=str)
    args = p.parse_args()

    CNNObj = CNN_ANN()
    CNNObj.build_model(learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS, nkerns=NKERNS)
    CNNObj.train_model(rgb=os.path.join(args.input_site, 'rgb.tif'), training_labels=os.path.join(args.input_site, 'tree_mask.tif'))
    CNNObj.validate_model(rgb=os.path.join(args.input_site, 'rgb.tif'), training_labels=os.path.join(args.input_site, 'tree_mask.tif'))
    CNNObj.extract_tree_model(rgb=os.path.join(args.input_site, 'rgb.tif'), output_labels=os.path.join(args.output, 'tree_labels.tif'))
    CNNObj.process_output(raster_data=os.path.join(args.input_site, 'rgb.tif'),
                          output_label=os.path.join(args.output, 'tree_labels.tif'), vector_data=os.path.join(args.input_site, 'power_ranch.sqlite'))

if __name__ == '__main__':
    main()
