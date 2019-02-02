using System;
using System.Collections.Generic;
using System.Text;

namespace MyNeuralNetwork.Tests.Interfaces
{
    interface INeuron
    {
        IEnumerable<double> Weights { get; set; }
        double Bias { get; set; }

        double Activation( IEnumerable<double> input );
        double Sigmoid( double input );
    }

}
