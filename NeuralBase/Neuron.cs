using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralBase
{
    class Neuron
    {
        public List<double> Weights { get; set; }
        public double Bias { get; set; }

        public double Aggregation(List<double> input)
        {
            throw new NotImplementedException();
        }

        public double Activation( List<double> input )
        {
            throw new NotImplementedException();
        }
    }
}
