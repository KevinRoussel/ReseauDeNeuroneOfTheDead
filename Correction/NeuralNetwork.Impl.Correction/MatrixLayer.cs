using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ITI.NeuralNetwork.Correction
{
    public class MatrixLayer
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
        public MatrixLayer( int size, int inputSize, Random generator )
        {
            Weights = Matrix<double>.Build.Dense( size, inputSize, ( i, j ) => generator.NextDouble() );
            Biases = Vector<double>.Build.Dense( size, ( i ) => generator.NextDouble() );
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

}
