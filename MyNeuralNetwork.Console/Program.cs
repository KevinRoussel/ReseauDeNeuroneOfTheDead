using NeuralNetwork.Correction;
using System;
using System.Linq;

namespace NeuralNetwork.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            Runner.Main();
        }

        static void ShowDatabase()
        {
            ImageProvider image = new ImageProvider();
            foreach(var el in image.ImageStream())
            {
                System.Console.WriteLine( el );
                System.Console.WriteLine( el.Label );

                System.Console.ReadLine();
            }
            image.Dispose();

        }

    }
}
