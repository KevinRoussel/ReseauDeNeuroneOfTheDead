using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.Correction
{
    public class Neuron
    {
        public List<double> Weights { get; set; }
        public double Bias { get; set; }

        public Neuron()
        {
            Weights = new List<double>();
        }

        /// <summary>
        /// Somme des Weight*Input + bias
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Aggregation( List<double> input )
        {
            return Weights.Zip( input, ( a, b ) => a * b ).Sum() + Bias;
            throw new NotImplementedException();
        }

        /// <summary>
        /// Valeur du neurone, sigmoid de l'aggregation
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Activation( List<double> input )
        {
            return Help.Sigmoid( Aggregation( input ) );
            throw new NotImplementedException();
        }
    }

}
