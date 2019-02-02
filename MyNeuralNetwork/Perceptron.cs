using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNeuralNetwork
{
    internal class Perceptron : Neuron
    {
        public Perceptron( List<double> weightsPerInput, double threshold ) : base(weightsPerInput, threshold)
        {
            
        }

        public override double Compute( IEnumerable<double> inputs )
        {
            if(inputs.Where(i=>i!=0 && i!=1).Count()>0)
                throw new ArgumentException("inputs must be 0 or 1", nameof(inputs));

            if( inputs.Count() != _weightsPerInput.Count() )
                throw new ArgumentException($"inputs count must match with {nameof(WeightsCount)}", nameof(inputs));

            return inputs.Zip( _weightsPerInput, ( input, _weight ) => input * _weight )
                    .Sum() + Bias > 0 ? 1 : 0;
        }


    }
}
