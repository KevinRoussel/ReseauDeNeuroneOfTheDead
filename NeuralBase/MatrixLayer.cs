using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralBase
{
    class MatrixLayer
    {
        public Matrix<double> Weights { get; private set; }
        public Vector<double> Biases { get; private set; }

        public int Size => throw new NotImplementedException();

        public MatrixLayer(int size, int inputSize)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Calcule le résultat de chaque neurone basé sur la valeur donnée
        /// </summary>
        /// <param name="data">input ou la valeur de la couche précédente</param>
        /// <returns>Valeur de la couche actuelle (un double de sortie par neurone)</returns>
        public Vector<double> Forward( Vector<double> data )
            => throw new NotImplementedException();

        /// <summary>
        /// Pour chaque neurone, calcul la somme des input*weight + biais
        /// </summary>
        /// <param name="data">input ou valeur de la couche précédente</param>
        /// <returns>Valeur temporaire de chaque neurone</returns>
        public Vector<double> Aggregation( Vector<double> data )
            => throw new NotImplementedException();

        /// <summary>
        /// Valeur finale de chaque neurone en appliquant la sigmoid
        /// </summary>
        /// <param name="x">Aggregat pour chaque neurone</param>
        /// <returns>Valeur finale du neurone</returns>
        public Vector<double> Activation( Vector<double> x )
            => throw new NotImplementedException();

        /// <summary>
        /// Dérivé de la valeur finale de chaque neurone
        /// </summary>
        /// <param name="x">Aggregat pour chaque neurone</param>
        /// <returns>Valeur dérivée du neurone</returns>
        public Vector<double> ActivationPrime( Vector<double> x )
        {
           throw new NotImplementedException();
        }
        public void UpdateWeighs( Matrix<double> gradient, double learningRate )
        {
            throw new NotImplementedException();
        }
        public void UpdateBiases( Vector<double> gradient, double learningRate )
        {
            throw new NotImplementedException();
        }

    }
}
