using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNeuralNetwork
{
    public class Network : INetwork
    {
        List<List<Neuron>> _network;
        double _learningRate;

        public int Result => _network.Last().Select((i,idx)=> new { content=i, idx=idx} ).OrderByDescending(i=>i.content.Value).First().idx;
        public string Values => _network.Last().Select( i => i.Value.ToString() ).Aggregate( ( a, b ) => a + "\n" +b );
        List<List<Neuron>> INetwork.InternalLayers => _network;

        public Network(int inputCount , double lr, params List<Neuron>[] layers)
        {
            _network = new List<List<Neuron>>() ;
            _learningRate = lr;
            var firstLayer = new List<Neuron>();
            for( var i = 0; i < inputCount; i++ ) firstLayer.Add( new InputNeuron() );
            _network.Add( firstLayer );

            _network.AddRange(layers.ToList());
            if( !CheckNetworkIntegrity( out List<Tuple<Neuron, int, int>> error ) )
            {
                Console.WriteLine( "Error" );
                throw new Exception();
            }
        }

        bool CheckNetworkIntegrity( out List<Tuple<Neuron, int, int>> error )
        {
            error = new List<Tuple<Neuron, int, int>>();
            for( var i = 1; i < _network.Count; i++ )
            {
                for( var j = 0; j < _network[i].Count; j++ )
                {
                    var neuron = _network[i][j];
                    if( neuron.Weights.Count() != _network[i - 1].Count )
                    {
                        error.Add( new Tuple<Neuron, int, int>( neuron, i, j ) );
                    }
                }
            }
            return error.Count == 0;
        }

        public IEnumerable<double> Run(IEnumerable<double> inputs)
        {
            if( inputs.Count() != _network[0].Count ) throw new Exception();

            // Layers processing
            for( var i=0; i<_network.Count; i++)
            {
                if(i==0) // First Layer process
                {
                    foreach( var el in _network[0].Zip( inputs, ( a, b ) => new Tuple<Neuron, double>( a, b ) ) )
                        el.Item1.Compute( new[] { el.Item2 } );
                }
                else // Other Layer process
                {
                    foreach(var el in _network[i])
                    {
                        el.Compute( _network[i-1].Select(neuron=>neuron.Value) );
                    }
                }
            }

            return _network.Last().Select(neuron=>neuron.Value);
        }


        public void Train( IList<double> input, IList<double> output )
        {
            Func<double,double,double> E = ( expected, current ) => 0.5 * Math.Pow( expected - current, 2 );
            Func<IEnumerable<double>, IEnumerable<double>, double> NeuronValue = ( h, weight ) => h.Zip( weight, ( a, b ) => a * b ).Sum();
            Func<double, double> LogisticFunction = ( z ) => 1 / (1 + Math.Exp( -z ));
            Func<double, double> DerivatedLogistic = ( z ) => LogisticFunction( z ) * (1 - LogisticFunction( z ));
            //Func<double, double> UpdateBias = (ref a) => { };

            //Run( input );

            foreach(var el in _network.Last().Zip(output, (a,b) => new Tuple<double,double>(a.Value,b)))
            {
                //E.Add(0.5*Math.Pow(el.Item2 - el.Item1 ,2));
            }





        }
    }
}
