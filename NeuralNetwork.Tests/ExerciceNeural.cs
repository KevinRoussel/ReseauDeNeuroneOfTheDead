using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Reflection;
using System.Threading.Tasks;
using System.Xml.Linq;
using NUnit.Framework;

namespace ITI.NeuralNetwork.Tests
{
    [TestFixture]
    public class ExerciceNeural
    {
        #region Exercice 1 - Neurone
        static Neuron SimpleNeuron()
            => new Neuron() { Bias = 0.5, Weights = new List<double>() { 0.1, 0.2, 0.3 } };

        // Exercice 1, Neurone
        [Test]
        void NeuronCreation()
        {
            Neuron n = SimpleNeuron();

            Assert.That( n.Bias, Is.EqualTo(0.5) );
            Assert.That(n.Weights[0]  , Is.EqualTo( 0.1 ) );
            Assert.That(n.Weights[1]  , Is.EqualTo( 0.2 ) );
            Assert.That(n.Weights[2]  , Is.EqualTo( 0.3 ) );
            Assert.That(n.Weights.Count , Is.EqualTo( 3 ) );
            Assert.Throws<ArgumentOutOfRangeException>( () => { var el = n.Weights[3]; } );
        }

        [Test]
        void NeuronAggregation()
        {
            Neuron n = SimpleNeuron();

            Assert.That( n.Aggregation( new List<double>() { 0.2, 0.8, 0.1 } ), Is.InRange(0.71, 0.72) );
            Assert.That( n.Aggregation( new List<double>() { 0.1, 0.1, 0.1 } ), Is.InRange(0.55, 0.56) );
        }

        [Test]
        void NeuronActivation()
        {
            Neuron n = SimpleNeuron();
            Assert.That( n.Activation( new List<double>() { 0.2, 0.8, 0.1 } ), Is.InRange( 0.67, 0.68) );
            Assert.That( n.Activation( new List<double>() { 0.1, 0.1, 0.1 } ), Is.InRange( 0.63, 0.64) );
        }
        #endregion

        #region Exercice 2 - Layer

        static Layer SimpleLayer()
            => new Layer() { Neurons = new List<Neuron>() { SimpleNeuron(), SimpleNeuron(), SimpleNeuron(), } };

        [Test]
        void LayerCreation()
        {
            Layer l = SimpleLayer();

            Assert.That(l.Neurons[0].Bias, Is.EqualTo(0.5) );
            Assert.That(l.Neurons[1].Bias, Is.EqualTo(0.5) );

            Assert.That( l.Neurons[0].Weights[0], Is.EqualTo( 0.1 ) );
            Assert.That( l.Neurons[0].Weights[1], Is.EqualTo( 0.2 ) );
            Assert.That( l.Neurons[0].Weights[2], Is.EqualTo( 0.3 ) );

            Assert.That( l.Neurons.Count, Is.EqualTo( 3 ) );
            Assert.That( l.Neurons[0].Weights.Count, Is.EqualTo( 3 ) );
            Assert.That( l.Neurons[2].Weights.Count, Is.EqualTo( 3 ) );
        }

        [Test]
        void LayerCompute()
        {
            Layer l = SimpleLayer();
            var el = l.Compute( new List<double>() { 0.3, 0.2, 0.9 } );
            var el2 = l.Compute( new List<double>() { 0.3, 0.2, 0.9 } );
            var el3 = l.Compute( new List<double>() { 1, 1, 1 } );
            var el4 = l.Compute( new List<double>() { 0, 0, 0 } );

            Assert.That( el[0]  , Is.InRange( 0.69, 0.70 ));
            Assert.That( el2[0] , Is.InRange( 0.69, 0.70 ));
            Assert.That( el3[0] , Is.InRange( 0.75, 0.76 ));
            Assert.That( el4[0] , Is.InRange( 0.62, 0.63 ));
        }

        #endregion

        #region Exercice 3 - Layer Matrix

        static MatrixLayer SimpleMatrixLayer()
            => new MatrixLayer( 3, 2, new Random( 1 ) );

        static Vector<double> BasicInput()
            => Vector<double>.Build.DenseOfArray( new[] { 0.2, 0.4 } );

        [Test]
        void MatrixLayerCreation()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            Assert.That( ml.Weights, Is.Not.Null );
            Assert.That( ml.Biases, Is.Not.Null );

            for(var i=0; i<ml.Weights.RowCount; i++)
            {
                for(var j=0; j<ml.Weights.ColumnCount; j++)
                {
                    Assert.That( ml.Weights[i, j], Is.InRange(0, 1) );
                }
            }
            for( var j = 0; j < ml.Biases.Count; j++ )
            {
                Assert.That( ml.Biases[j], Is.InRange(0, 1) );
            }
        }

        [Test]
        void MatrixForward()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Forward( BasicInput() );
            Assert.That(result[0], Is.InRange(0.67, 0.68) );
            Assert.That(result[1], Is.InRange(0.77, 0.78) );
            Assert.That(result[2], Is.InRange(0.59, 0.60) );
        }

        [Test]
        void MatrixAggregation()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Aggregation( BasicInput() );
            Assert.That(result[0], Is.InRange(0.71, 0.72 ));
            Assert.That(result[1], Is.InRange(1.22, 1.23 ));
            Assert.That(result[2], Is.InRange( 0.36, 0.37));
        }

        [Test]
        void MatrixActivation()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.Activation( BasicInput() );
            Assert.That(result[0], Is.InRange(0.54, 0.55 ));
            Assert.That(result[1], Is.InRange( 0.59, 0.60) );
        }

        [Test]
        void MatrixActivationPrime()
        {
            MatrixLayer ml = SimpleMatrixLayer();

            var result = ml.ActivationPrime( BasicInput() );
            Assert.That(result[0], Is.InRange(0.24, 0.25 ));
            Assert.That(result[1], Is.InRange( 0.24, 0.25) );
        }

        #endregion

        #region Exercice 4 Matrix Network

        static MatrixNetwork SimpleMatrixNetwork()
            => new MatrixNetwork( 2 );
        static MatrixNetwork NetworkWithLayer()
            => SimpleMatrixNetwork()
                .AddLayer( 3, new Random(1) )
                .AddLayer(2, new Random( 2 ) );

        [Test]
        void NetworkCreation()
        {
            var net = SimpleMatrixNetwork();
            Random r = new Random( 1 );
            Assert.That( net.Layers, Is.Empty);

            net.AddLayer( 2, r );
            Assert.That( net.Layers, Is.EqualTo(1));

            net.AddLayer( 3, r );
            Assert.That( net.Layers.Count(), Is.EqualTo(2) );
        }

        [Test]
        void NetworkFeedForward()
        {
            var net = NetworkWithLayer();
            var el = net.FeedForward( BasicInput() );

            Assert.That( el[0], Is.InRange(0.81, 0.82) );
            Assert.That( el[1], Is.InRange(0.84, 0.85) );
        }

        [Test]
        void NetworkOutputDelta()
        {
            var net = NetworkWithLayer();
            Vector<double> A = Vector<double>.Build.DenseOfArray( new double[] { 0.1, 0.3 } );
            Vector<double> B = Vector<double>.Build.DenseOfArray( new double[] { 1, 0 } );
            var el = net.GetOutputDelta( A, B );

            Assert.That(el[0], Is.EqualTo(-0.9));
            Assert.That(el[1], Is.EqualTo(0.3) );
        }

        [Test]
        void NetworkPredict()
        {
            var net = NetworkWithLayer();
            var input = BasicInput();
            var result = net.Predict(input, out List<double> t );

            Assert.That( result, Is.EqualTo(1) );
            Assert.That( t[0], Is.InRange(0.81, 0.82 ));
            Assert.That( t[1], Is.InRange( 0.84, 0.85) );
        }

        [Test]
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

                Assert.That(performance, Is.InRange( 6000, total));
            }
        }

        #endregion

    }
}
