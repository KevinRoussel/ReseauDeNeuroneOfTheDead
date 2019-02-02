using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace Transpilation
{
    public class Program
    {
        public static void Main()
        {
            // Adaptation du modèle en matrice
            var imageProvider = new ImageProvider().ImageStream().ToList();
            List<Vector<double>> X = new List<Vector<double>>( imageProvider.Select( i => i.SampledPixels ).Select( i => Vector<double>.Build.DenseOfEnumerable( i ) ) );
            List<int> Y = imageProvider.Select( i => i.Label ).ToList();

            var net = new Network( 784 );
            net.AddLayer( 200 );
            net.AddLayer( 10 );

            Console.WriteLine( $"Start Training ... {DateTime.Now}" );
            net.Train( X, Y );
            Console.WriteLine( $"Training Fnished ... {DateTime.Now}" );

            Console.WriteLine( $"Evaluatation ..." );

            foreach(var el in X.Zip(Y, ( input, expected ) => new { input, expected } ))
            {
                Console.WriteLine($"Expected : {el.expected}, Output :{net.Predict( el.input)} ");
                Console.ReadLine();
            }
            Console.ReadLine();

        }

    };

    public static class Help
    {
        public static double Sigmoid( double x ) => 1.0 / (1.0 + Math.Exp( -x ));
        public static double SigmoidPrime( double x ) => Help.Sigmoid( x ) * (1.0 - Sigmoid( x ));
        /// <summary>
        /// Convertit un entier en vecteur "one-hot".
        /// to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
        /// </summary>
        /// <param name="y"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static Vector<double> ToOneHot( double y, int k ) => Vector<double>.Build.Dense( k, ( index ) => index == y ? 1.0 : 0.0 );

        public static void Shuffle<T>( this IList<T> list, int seed )
        {
            Random rng = new Random( seed );
            int n = list.Count;
            while( n > 1 )
            {
                n--;
                int k = rng.Next( n + 1 );
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }

    public class Layer
    {
        /// <summary>
        /// Matrice contenant contenant pour chaque neurone, les poids de chaque entré
        /// Fonctionne grossièrement comme un tableau [,]
        /// Weights[i][j] => neurone numéro [i] au poid numéro[j]
        /// </summary>
        public Matrix<double> Weights { get; private set; }
        /// <summary>
        /// Matrice contenant contenant pour chaque neurone, le biais
        /// Fonctionne grossièrement comme un tableau []
        /// Bias[i] => biais du neurone numéro [i]
        /// </summary>
        public Vector<double> Biases { get; private set; }
        /// <summary>
        /// Nombre de neurone contenu dans le layer
        /// </summary>
        public int Size => Weights.RowCount;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="size">Nombre de neurone sur cette couche</param>
        /// <param name="inputSize">Nombre de neurone sur la couche précédente</param>
        public Layer( int size, int inputSize )
        {
            Random r = new Random( 1 );
            Weights = Matrix<double>.Build.Dense( size, inputSize, ( i, j ) => r.NextDouble() );
            Biases = Vector<double>.Build.Dense( size, ( i ) => r.NextDouble() );
        }

        /// <summary>
        /// Calcule le résultat de chaque neurone basé sur la valeur donnée
        /// </summary>
        /// <param name="data">input ou la valeur de la couche précédente</param>
        /// <returns>Valeur de la couche actuelle (un double de sortie par neurone)</returns>
        public Vector<double> Forward( Vector<double> data )
            => Activation( Aggregation( data ) );

        /// <summary>
        /// Pour chaque neurone, calcul la somme des input*weight + biais
        /// </summary>
        /// <param name="data">input ou valeur de la couche précédente</param>
        /// <returns>Valeur temporaire de chaque neurone</returns>
        public Vector<double> Aggregation( Vector<double> data )
            => Weights * data + Biases;

        /// <summary>
        /// Valeur finale de chaque neurone en appliquant la sigmoid
        /// </summary>
        /// <param name="x">Aggregat pour chaque neurone</param>
        /// <returns>Valeur finale du neurone</returns>
        public Vector<double> Activation( Vector<double> x )
            => Vector<double>.Build.DenseOfEnumerable( x.Select( i => Help.Sigmoid( i ) ) );


        
        public Vector<double> ActivationPrime( Vector<double> x )
        {
            return Vector<double>.Build.DenseOfEnumerable( x.Select( i => Help.SigmoidPrime( i ) ) );
        }
        public void UpdateWeighs( Matrix<double> gradient, double learningRate )
        {
            Weights -= learningRate * gradient;
        }
        public void UpdateBiases( Vector<double> gradient, double learningRate )
        {
            Biases -= learningRate * gradient;
        }

    }

    public class Network
    {
        /// <summary>
        /// Nombre d'input que l'on a en entré
        /// </summary>
        int inputDim;

        /// <summary>
        /// Les différentes couches de neurones du réseaux (hors input, output compris)
        /// </summary>
        List<Layer> layers;

        /// <summary>
        /// Création d'un réseau de neurone
        /// </summary>
        /// <param name="inputDim">nombre d'entré pour le réseau</param>
        public Network( int inputDim )
        {
            this.inputDim = inputDim;
            layers = new List<Layer>();
        }

        /// <summary>
        /// Ajoute un layer au network de la taille renseigné
        /// </summary>
        /// <param name="size">Nombre de neurone du layer créé</param>
        public void AddLayer( int size )
            => layers.Add( new Layer( size, layers.Count>0 ? layers.Last().Size: inputDim ) );

        /// <summary>
        /// Récupère l'entré et propage l'information au travers du réseau
        /// </summary>
        /// <param name="inputData">Input</param>
        /// <returns>Valeur de la couche de sortie</returns>
        public Vector<double> FeedForward( Vector<double> inputData )
        {
            var activation = inputData;
            foreach( var layer in layers )
            {
                activation = layer.Forward( activation );
            }

            return activation;
        }

        /// <summary>
        /// Annonce le retour du réseaux basé sur l'input fourni
        /// </summary>
        /// <param name="inputData">Input</param>
        /// <returns>Réponse du réseau</returns>
        public int Predict( Vector<double> inputData )
            => FeedForward( inputData ).Select( ( i, idx ) => (idx: idx, i: i) ).OrderByDescending( i => i.i ).First().idx;

        /// <summary>
        /// Reçoit plusieurs images, et calcul le taux de bonne réponse du réseau
        /// </summary>
        /// <param name="images">List of image (List<double>) </param>
        /// <param name="responses">List of expected result</param>
        public double Evaluate( Matrix<double> images, Vector<int> responses )
            => images.AsRowArrays()
                .Zip( responses, ( image, expected ) => new { image, expected } )
                .Select( i => Predict( Vector<double>.Build.DenseOfArray(i.image) ) == i.expected ? 1 : 0 )
                .Sum() / images.RowCount;
           


        /// <summary>
        /// 
        /// </summary>
        /// <param name="X">Expected ?</param>
        /// <param name="Y">Result</param>
        /// <param name="steps">Iteraion</param>
        /// <param name="learningRate"></param>
        /// <param name="batchSize"></param>
        public void Train( List<Vector<double>> X, List<int> Y, int steps = 30, double learningRate = 0.3, int batchSize = 10 )
        {
            var n = Y.Count;
            for( var i = 0; i < steps; i++ )
            {
                X.Shuffle( i );
                Y.Shuffle( i );
                for( var batchStart = 0; batchStart < n; batchStart += batchSize )
                {
                    var X_batch = X.Skip( batchStart ).Take( batchSize ).ToList();
                    var Y_batch = Y.Skip( batchStart ).Take( batchSize ).ToList();
                    TrainBatch( X_batch, Y_batch, learningRate );
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="X"></param>
        /// <param name="Y"></param>
        /// <param name="learningRate"></param>
        public void TrainBatch( List<Vector<double>> X, List<int> Y, double learningRate )
        {
            List<Matrix<double>> weight_gradient = new List<Matrix<double>>( layers.Select( i => Matrix<double>.Build.Dense( i.Weights.RowCount, i.Weights.ColumnCount, 0 ) ) );
            List<Vector<double>> bias_gradient = new List<Vector<double>>( layers.Select( i => Vector<double>.Build.Dense( i.Size, 0 ) ) );

            foreach( var (x, y) in X.Zip( Y, ( @x, @y ) => (@x, @y) ) )
            {
                var (new_weight_gradient, new_bias_gradient) = BackProp( x, y );
                weight_gradient = weight_gradient.Zip( new_weight_gradient, ( wg, nwg ) => wg + nwg ).ToList();
                bias_gradient = bias_gradient.Zip( new_bias_gradient, ( bg, nbg ) => bg + nbg ).ToList();
            }
            var avg_weight_gradient = weight_gradient.Select( wg => wg / Y.Count ).ToList();
            var avg_bias_gradient = bias_gradient.Select( bg => bg / Y.Count ).ToList();

            foreach( var el in
                this.layers.Zip( avg_weight_gradient, ( a, b ) => new Tuple<Layer,Matrix<double>>(a, b) )
                .Zip( avg_bias_gradient, ( el, c ) => new { layer = el.Item1, weight_gradient= el.Item2, bias_gradient= c}))
            {
                el.layer.UpdateWeighs( el.weight_gradient, learningRate );
                el.layer.UpdateBiases( el.bias_gradient, learningRate );
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x">ImageStream</param>
        /// <param name="y">Expected output</param>
        /// <returns></returns>
        public (List<Matrix<double>>, List<Vector<double>>) BackProp( Vector<double> x, int y )
        {
            List<Vector<double>> aggregations = new List<Vector<double>>();
            List<Vector<double>> activations = new List<Vector<double>>() { x };
            List<Vector<double>> activationPrimes = new List<Vector<double>>();

            Vector<double> activation = x;
            Vector<double> aggregation = null;
            Vector<double> activationPrime = null;

            // Propagation pour obtenir la sortie
            foreach( var layer in layers )
            {
                aggregation = layer.Aggregation( activation );
                activation = layer.Activation( aggregation );
                activationPrime = layer.ActivationPrime( aggregation );

                aggregations.Add( aggregation );
                activations.Add( activation );
                activationPrimes.Add( activationPrime );
            }

            // Calculons la valeur delta (δ) pour la dernière couche
            // en appliquant les équations détaillées plus haut.
            Vector<double> target = Help.ToOneHot( y, 10 );
            Vector<double> delta = GetOutputDelta( activations.Last(), target );
            List<Vector<double>> deltas = new List<Vector<double>>() { delta };

            // Phase de rétropropagation pour calculer les deltas de chaque
            // couche
            // On utilise une implémentation vectorielle des équations.
            for( var l = layers.Count - 2; l >= 0; l-- )
            {
                var nextLayer = layers[l + 1];
                delta = activationPrime[l] * nextLayer.Weights.Transpose() * delta;
                deltas.Add( delta );
            }

            // Nous sommes parti de l'avant-dernière couche pour remonter vers
            // la première. deltas[0] contient le delta de la dernière couche.
            // Nous l'inversons pour faciliter la gestion des indices plus tard.
            deltas.Reverse();

            // On utilise maintenant les deltas pour calculer les gradients.
            List<Matrix<double>> weight_gradient = new List<Matrix<double>>();
            var bias_gradient = new List<Vector<double>>();
            for( var l = 0; l < layers.Count; l++ )
            {
                // Notez que l'indice des activations est « décalé », puisque
                // activation[0] contient l'entrée (x), et pas l'activation de
                // la première couche.
                var prev_activation = activations[l];
                weight_gradient.Add( Vector<double>.OuterProduct( deltas[l], prev_activation ) );
                bias_gradient.Add( deltas[l] );
            }
            return (weight_gradient, bias_gradient);
        }

        /// <summary>
        /// Calcule le delta de la couche donné en utilisant les valeurs activation et la cible
        /// </summary>
        /// <param name="aggregations"></param>
        /// <param name="activations"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public Vector<double> GetOutputDelta( Vector<double> activations, Vector<double> target )
        {
            return activations - target;
        }

    }

}
