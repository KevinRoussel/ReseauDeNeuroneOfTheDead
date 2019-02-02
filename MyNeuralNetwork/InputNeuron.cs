using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNeuralNetwork
{
    internal class InputNeuron : Neuron
    {
        public InputNeuron( ) : base( new double[] { 1 }, 0 ) { }

        public override double Compute( IEnumerable<double> inputs )
            => Value = inputs.First();
    }
}
