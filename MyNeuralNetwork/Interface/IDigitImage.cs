using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNeuralNetwork
{
    public interface IDigitImage
    {
        IEnumerable<double> SampledPixels { get; }
        int Label { get; }
        IEnumerable<byte> Pixels();

    }
}
