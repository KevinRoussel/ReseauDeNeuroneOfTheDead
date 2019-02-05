using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITI.NeuralNetwork.Correction
{
    public class Runner
    {
        public static void Main()
        {
            Console.WriteLine( "##### Neural Network of the Dead #####" );

            // Adaptation du modèle en matrice
            var imageProvider = new ImageProvider().ImageStream().ToList();
            List<Vector<double>> X = new List<Vector<double>>( imageProvider.Select( i => i.SampledPixels ).Select( i => Vector<double>.Build.DenseOfEnumerable( i ) ) );
            List<int> Y = imageProvider.Select( i => i.Label ).ToList();
            Random r = new Random( 1 );
            var net = new MatrixNetwork( 784 );
            net.AddLayer( 200, r );
            net.AddLayer( 10, r );
            Console.WriteLine( $"Start Training ... {DateTime.Now}" );
            net.Train( X, Y );
            Console.WriteLine( $"Training Finished ... {DateTime.Now}" );

            Console.WriteLine( "Writing Neural Network Snapshot..." );
            string json = JsonConvert.SerializeObject( net.ToFlatNetwork() );
            System.IO.File.WriteAllText( @"E:\neuralNetwork.json", json );
            Console.WriteLine( "Writing Done ..." );

            Console.WriteLine( $"Evaluation ..." );
            foreach( var el in X.Zip( Y, ( input, expected ) => new { input, expected } ) )
            {
                List<double> output = null;
                Console.WriteLine( $"Expected : {el.expected}, Output :{net.Predict( el.input, out output )} " );
                foreach( var outputline in output.Select( ( i, idx ) => new { i, idx } ) )
                {
                    Console.WriteLine( $"idx {outputline.idx} => {outputline.i}" );
                }
                Console.ReadLine();
            }

            Console.ReadLine();

        }

    };

}
