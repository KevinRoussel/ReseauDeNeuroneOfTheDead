using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Correction
{
    public static class Help
    {
        public class FlatNetwork
        {
            public FlatNetwork()
            {
                Layers = new List<FlatLayer>();
            }
            public List<FlatLayer> Layers { get; private set; }
        }
        public class FlatLayer
        {
            public FlatLayer()
            {
                Neurons = new List<FlatNeuron>();
            }
            public List<FlatNeuron> Neurons { get; private set; }
        }
        public class FlatNeuron
        {
            public FlatNeuron()
            {
                Weights = new List<double>();
                Bias = 0;
            }
            public List<double> Weights { get; private set; }
            public double Bias { get; set; }
        }

        public static double Sigmoid( double x ) => 1.0 / (1.0 + Math.Exp( -x ));
        public static double SigmoidPrime( double x ) => Help.Sigmoid( x ) * (1.0 - Sigmoid( x ));
        /// <summary>
        /// Convertit un entier en vecteur "one-hot".
        /// to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
        /// </summary>
        /// <param name="y"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static Vector<double> ToOneHot( double y, int k ) => Vector<double>.Build.Dense( k, ( index ) => index == y ? 1.0 : 0.0 );

        public static void Shuffle<T>( this IList<T> list, int seed )
        {
            Random rng = new Random( seed );
            int n = list.Count;
            while( n > 1 )
            {
                n--;
                int k = rng.Next( n + 1 );
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static FlatNetwork ToFlatNetwork( this MatrixNetwork @this )
        {
            var net = new FlatNetwork();
            foreach( var layer in @this.Layers )
            {
                var flatLayer = new FlatLayer();
                for( var i = 0; i < layer.Weights.RowCount; i++ )
                {
                    var el = new { w = layer.Weights.Row( i ), b = layer.Biases[i] };

                    var flatNeuron = new FlatNeuron();
                    flatNeuron.Weights.AddRange( el.w );
                    flatNeuron.Bias = el.b;

                    flatLayer.Neurons.Add( flatNeuron );
                }

                net.Layers.Add( flatLayer );
            }
            return net;
        }

    }
}
