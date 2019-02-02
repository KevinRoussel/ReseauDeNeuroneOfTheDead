using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNeuralNetwork
{
    public abstract class Neuron : INeuron
    {
        protected List<double> _weightsPerInput;
        protected int WeightsCount => _weightsPerInput.Count;

        public double Value { get; set; }
        public double Delta { get; set; }
        public double Bias { get; set; }
        public IList<double> Weights { get => _weightsPerInput; }

        public Neuron(IEnumerable<double> weightsPerInput, double bias )
        {
            _weightsPerInput = weightsPerInput.ToList();
            Bias = bias;
        }

        public abstract double Compute( IEnumerable<double> inputs );

        public override string ToString()
        {
            return $"b: {Bias} w1: {Weights[0]}";
            return base.ToString();
        }
    }
}
