using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;


//using ITI.NeuralNetwork.Correction;
using ITI.NeuralNetwork;

namespace NeuralNetwork.Tests
{
    public class ExerciceNeural
    {

    #region Exercice 1 - Neurone
        static Neuron SimpleNeuron()
            => new Neuron() { Bias = 0.5, Weights = new List<double>() { 0.1, 0.2, 0.3 } };

        // Exercice 1, Neurone
        [Fact]
        void NeuronCreation()
        {
            Neuron n = SimpleNeuron();

            Assert.Equal( 0.5, n.Bias );
            Assert.Equal( 0.1, n.Weights[0]);
            Assert.Equal( 0.2, n.Weights[1]);
            Assert.Equal( 0.3, n.Weights[2]);
            Assert.Equal( 3, n.Weights.Count );
            Assert.Throws<ArgumentOutOfRangeException>( () => { var el = n.Weights[3]; } );
        }

        [Fact]
        void NeuronAggregation()
        {
            Neuron n = SimpleNeuron();

            Assert.True( 0.71 < n.Aggregation( new List<double>() { 0.2, 0.8, 0.1 } ) );
            Assert.True( 0.72 > n.Aggregation( new List<double>() { 0.2, 0.8, 0.1 } ) );

            Assert.True( 0.55 < n.Aggregation( new List<double>() { 0.1, 0.1, 0.1 } ) );
            Assert.True( 0.57 > n.Aggregation( new List<double>() { 0.1, 0.1, 0.1 } ) );
        }

        [Fact]
        void NeuronActivation()
        {
            Neuron n = SimpleNeuron();

            Assert.True( 0.67 < n.Activation( new List<double>() { 0.2, 0.8, 0.1 } ) );
            Assert.True( 0.68 > n.Activation( new List<double>() { 0.2, 0.8, 0.1 } ) );

            Assert.True( 0.63 < n.Activation( new List<double>() { 0.1, 0.1, 0.1 } ) );
            Assert.True( 0.64 > n.Activation( new List<double>() { 0.1, 0.1, 0.1 } ) );
        }
        #endregion

    #region Exercice 2 - Layer

        static Layer SimpleLayer()
            => new Layer() { Neurons = new List<Neuron>() { SimpleNeuron(), SimpleNeuron(), SimpleNeuron(), } };

        [Fact]
        void LayerCreation()
        {
            Layer l = SimpleLayer();

            Assert.Equal( 0.5, l.Neurons[0].Bias );
            Assert.Equal( 0.5, l.Neurons[1].Bias );

            Assert.Equal( 0.1, l.Neurons[0].Weights[0]);
            Assert.Equal( 0.2, l.Neurons[0].Weights[1]);
            Assert.Equal( 0.3, l.Neurons[0].Weights[2]);

            Assert.Equal( 3, l.Neurons.Count );
            Assert.Equal( 3, l.Neurons[0].Weights.Count );
            Assert.Equal( 3, l.Neurons[2].Weights.Count );
        }

        [Fact]
        void LayerCompute()
        {
            Layer l = SimpleLayer();
            var el = l.Compute( new List<double>() { 0.3, 0.2, 0.9 } );
            var el2 = l.Compute( new List<double>() { 0.3, 0.2, 0.9 } );
            var el3 = l.Compute( new List<double>() { 1, 1, 1 } );
            var el4 = l.Compute( new List<double>() { 0, 0, 0 } );

            Assert.True( el[0] > 0.69 );
            Assert.True( el[0] < 0.70 );

            Assert.True( el2[0] > 0.69 );
            Assert.True( el2[0] < 0.70 );

            Assert.True( el3[0] > 0.75 );
            Assert.True( el3[0] < 0.76 );

            Assert.True( el4[0] > 0.62 );
            Assert.True( el4[0] < 0.63 );
        }

        #endregion

    #region Exercice 3 - Layer Matrix

        static MatrixLayer SimpleMatrixLayer()
            => new MatrixLayer( 3, 2, new Random( 1 ) );

        static Vector<double> BasicInput()
            => Vector<double>.Build.DenseOfArray( new[] { 0.2, 0.4 } );

        [Fact]
        void MatrixLayerCreation()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            Assert.True( ml.Weights != null);
            Assert.True( ml.Biases != null);

            for(var i=0; i<ml.Weights.RowCount; i++)
            {
                for(var j=0; j<ml.Weights.ColumnCount; j++)
                {
                    Assert.True(ml.Weights[i,j] >= 0);
                    Assert.True(ml.Weights[i,j] <= 1);
                }
            }
            for( var j = 0; j < ml.Biases.Count; j++ )
            {
                Assert.True( ml.Biases[j] >= 0 );
                Assert.True( ml.Biases[j] <= 1 );
            }
        }

        [Fact]
        void MatrixForward()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Forward( BasicInput() );
            Assert.True(0.67 < result[0]);
            Assert.True(0.68 > result[0]);

            Assert.True( 0.77 < result[1] );
            Assert.True( 0.78 > result[1] );

            Assert.True( 0.59 < result[2] );
            Assert.True( 0.60 > result[2] );
        }

        [Fact]
        void MatrixAggregation()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Aggregation( BasicInput() );
            Assert.True( 0.71 < result[0] );
            Assert.True( 0.72 > result[0] );

            Assert.True( 1.22 < result[1] );
            Assert.True( 1.23 > result[1] );

            Assert.True( 0.36 < result[2] );
            Assert.True( 0.37 > result[2] );
        }

        [Fact]
        void MatrixActivation()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Activation( BasicInput() );
            Assert.True( 0.54 < result[0] );
            Assert.True( 0.55 > result[0] );

            Assert.True( 0.59 < result[1] );
            Assert.True( 0.60 > result[1] );
        }

        [Fact]
        void MatrixActivationPrime()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.ActivationPrime( BasicInput() );
            Assert.True( 0.24 < result[0] );
            Assert.True( 0.25 > result[0] );

            Assert.True( 0.24 < result[1] );
            Assert.True( 0.25 > result[1] );
        }

        #endregion

        #region Exercice 4 Matrix Network

        static MatrixNetwork SimpleMatrixNetwork()
            => new MatrixNetwork( 2 );
        static MatrixNetwork NetworkWithLayer()
            => SimpleMatrixNetwork()
                .AddLayer( 3, new Random(1) )
                .AddLayer(2, new Random( 2 ) );

        [Fact]
        void NetworkCreation()
        {
            var net = SimpleMatrixNetwork();
            Random r = new Random( 1 );
            Assert.Empty( net.Layers);

            net.AddLayer( 2, r );
            Assert.Single( net.Layers);

            net.AddLayer( 3, r );
            Assert.Equal( 2, net.Layers.Count() );

        }

        [Fact]
        void NetworkFeedForward()
        {
            var net = NetworkWithLayer();
            var el = net.FeedForward( BasicInput() );

            Assert.True( 0.81 < el[0] );
            Assert.True( 0.82 > el[0] );

            Assert.True( 0.84 < el[1] );
            Assert.True( 0.85 > el[1] );
        }

        [Fact]
        void NetworkOutputDelta()
        {
            var net = NetworkWithLayer();
            Vector<double> A = Vector<double>.Build.DenseOfArray( new double[] { 0.1, 0.3 } );
            Vector<double> B = Vector<double>.Build.DenseOfArray( new double[] { 1, 0 } );
            var el = net.GetOutputDelta( A, B );

            Assert.Equal( -0.9 , el[0]);
            Assert.Equal( 0.3, el[1]);
        }

        [Fact]
        void NetworkPredict()
        {
            var net = NetworkWithLayer();
            var input = BasicInput();
            var result = net.Predict(input, out List<double> t );

            Assert.Equal( 1, result );

            Assert.True( 0.81 < t[0] );
            Assert.True( 0.82 > t[0] );

            Assert.True( 0.84 < t[1] );
            Assert.True( 0.85 > t[1] );
        }

        [Fact]
        void TrainWorks()
        {
            var imageProvider = new ImageProvider(
                    @"F:\#FICHIER\ITI-Train\Data\train-labels.idx1-ubyte",
                    @"F:\#FICHIER\ITI-Train\Data\train-images.idx3-ubyte")
                    .ImageStream();

            List<(Vector<double>,int)> tmp = imageProvider
                    .Select( i => new { pixels = i.SampledPixels, i.Label } )
                    .Select( i => (Vector<double>.Build.DenseOfEnumerable( i.pixels ), i.Label )).ToList();
            var X = tmp.Select( i => i.Item1 ).ToList();
            var Y = tmp.Select( i => i.Item2 ).ToList();

             imageProvider.Select( i => i.Label ).ToList();
            Random r = new Random( 1 );
            var net = new MatrixNetwork( 784 );
            net.AddLayer( 200, r );
            net.AddLayer( 10, r );
            net.Train( X, Y, 15, 0.3, 10);

            int performance = 0;
            int total = 0;
            foreach( var el in X.Zip( Y, ( input, expected ) => new { input, expected } ) )
            {
                total++;
                var result = net.Predict( el.input, out List<double> output );
                if( el.expected == result ) performance++;
            }
            Assert.True( performance / total > 0.5 );

        }

        #endregion

    }
}
