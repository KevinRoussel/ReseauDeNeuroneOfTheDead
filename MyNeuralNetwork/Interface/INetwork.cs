using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNeuralNetwork
{
    interface INetwork
    {
        List<List<Neuron>> InternalLayers { get; }
        IEnumerable<double> Run( IEnumerable<double> inputs );
        void Train( IList<double> input, IList<double> expectedOutput );


    }
}
