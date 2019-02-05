using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

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

            Assert.InRange( n.Aggregation( new List<double>() { 0.2, 0.8, 0.1 } ), 0.71, 0.72 );
            Assert.InRange( n.Aggregation( new List<double>() { 0.1, 0.1, 0.1 } ), 0.55, 0.56 );
        }

        [Fact]
        void NeuronActivation()
        {
            Neuron n = SimpleNeuron();

            Assert.InRange( n.Activation( new List<double>() { 0.2, 0.8, 0.1 } ), 0.67, 0.68 );
            Assert.InRange( n.Activation( new List<double>() { 0.1, 0.1, 0.1 } ), 0.63, 0.64 );
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

            Assert.InRange( el[0] , 0.69, 0.70 );
            Assert.InRange( el2[0] , 0.69, 0.70 );
            Assert.InRange( el3[0], 0.75, 0.76 );
            Assert.InRange( el4[0], 0.62, 0.63 );
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

            Assert.NotNull( ml.Weights );
            Assert.NotNull( ml.Biases );

            for(var i=0; i<ml.Weights.RowCount; i++)
            {
                for(var j=0; j<ml.Weights.ColumnCount; j++)
                {
                    Assert.InRange( ml.Weights[i, j], 0, 1 );
                }
            }
            for( var j = 0; j < ml.Biases.Count; j++ )
            {
                Assert.InRange( ml.Biases[j], 0, 1 );
            }
        }

        [Fact]
        void MatrixForward()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Forward( BasicInput() );
            Assert.InRange(result[0], 0.67, 0.68 );
            Assert.InRange(result[1], 0.77, 0.78 );
            Assert.InRange(result[2], 0.59, 0.60 );
        }

        [Fact]
        void MatrixAggregation()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Aggregation( BasicInput() );
            Assert.InRange(result[0], 0.71, 0.72 );
            Assert.InRange(result[1], 1.22, 1.23 );
            Assert.InRange(result[2], 0.36, 0.37 );
        }

        [Fact]
        void MatrixActivation()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Activation( BasicInput() );
            Assert.InRange(result[0], 0.54, 0.55 );
            Assert.InRange(result[1], 0.59, 0.60 );
        }

        [Fact]
        void MatrixActivationPrime()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.ActivationPrime( BasicInput() );
            Assert.InRange(result[0], 0.24, 0.25 );
            Assert.InRange(result[1], 0.24, 0.25 );
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

            Assert.InRange( el[0], 0.81, 0.82 );
            Assert.InRange( el[1], 0.84, 0.85 );
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
            Assert.InRange( t[0], 0.81, 0.82 );
            Assert.InRange( t[1], 0.84, 0.85 );
        }

        [Fact]
        void TrainWorks()
        {
            using( var ip = new ImageProvider())
            {
                var imageProvider = ip.ImageStream();
                List<(Vector<double>, int)> tmp = imageProvider
                        .Select( i => new { pixels = i.SampledPixels, i.Label } )
                        .Select( i => (Vector<double>.Build.DenseOfEnumerable( i.pixels ), i.Label) ).ToList();
                var X = tmp.Select( i => i.Item1 ).ToList();
                var Y = tmp.Select( i => i.Item2 ).ToList();

                 imageProvider.Select( i => i.Label ).ToList();
                Random r = new Random( 1 );
                var net = new MatrixNetwork( 784 );
                net.AddLayer( 200, r );
                net.AddLayer( 10, r );
                net.Train( X, Y );

                int performance = 0;
                int total = 0;
                foreach( var el in X.Zip( Y, ( input, expected ) => new { input, expected } ) )
                {
                    total++;
                    var result = net.Predict( el.input, out List<double> output );
                    if( el.expected == result ) performance++;
                }

                Assert.InRange(performance, 6000, total);
            }
        }

        #endregion

    }
}
