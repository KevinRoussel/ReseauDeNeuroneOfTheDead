using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNeuralNetwork
{
    public class SigmoidNeuron : Neuron
    {
        public SigmoidNeuron( IEnumerable<double> weightsPerInput, double bias ) : base( weightsPerInput, bias )
        {

        }

        public override double Compute( IEnumerable<double> inputs )
        {
            if( inputs.Count() != _weightsPerInput.Count() )
                throw new ArgumentException( $"inputs count must match with {nameof( WeightsCount )}", nameof( inputs ) );

            var aggregation = inputs.Zip( _weightsPerInput, ( a, b ) => a * b ).Sum();
            return Value = Sigmoid(aggregation + Bias);
        }

        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

    }
}
