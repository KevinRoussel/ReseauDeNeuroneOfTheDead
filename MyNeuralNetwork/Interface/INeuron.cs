using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNeuralNetwork
{
    interface INeuron
    {
        double Value { get; }
        double Bias { get; }
        IList<double> Weights { get; }

        double Compute( IEnumerable<double> inputs );
    }
}
