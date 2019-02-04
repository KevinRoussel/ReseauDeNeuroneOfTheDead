using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.Correction
{
    public class Layer
    {
        public List<Neuron> Neurons { get; set; }

        public List<double> Compute( List<double> input )
        {
            return Neurons.Select( i => i.Activation( input ) ).ToList();
            throw new NotImplementedException();
        }
    }

}
