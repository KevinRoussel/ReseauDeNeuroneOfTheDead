using Transpilation;
using System;
using System.Linq;

namespace MyNeuralNetwork.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            Transpilation.Program.Main();
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
