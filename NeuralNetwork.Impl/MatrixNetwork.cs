using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace ITI.NeuralNetwork
{
    public class MatrixNetwork
    {
        public int InputSize { get; private set; }
        public List<MatrixLayer> Layers { get; private set; }

        public MatrixNetwork(int inputSize)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Ajoute un layer de [Size] neurone
        /// </summary>
        /// <param name="size"></param>
        public MatrixNetwork AddLayer(int size, Random r)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Récupère l'entré et propage l'information au travers du réseau
        /// </summary>
        /// <param name="inputData">Input</param>
        /// <returns>Valeur de la couche de sortie</returns>
        public Vector<double> FeedForward( Vector<double> inputData )
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Annonce le retour du réseaux basé sur l'input fourni
        /// </summary>
        /// <param name="inputData">Input</param>
        /// <returns>Réponse du réseau</returns>
        public int Predict( Vector<double> inputData, out List<double> outputNeural )
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Reçoit plusieurs images, et calcul le taux de bonne réponse du réseau
        /// </summary>
        /// <param name="images">List of image (List<double>) </param>
        /// <param name="responses">List of expected result</param>
        public double Evaluate( Matrix<double> images, Vector<int> responses )
        {
            throw new NotImplementedException();
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="X">Expected ?</param>
        /// <param name="Y">Result</param>
        /// <param name="steps">Iteraion</param>
        /// <param name="learningRate"></param>
        /// <param name="batchSize"></param>
        public void Train( List<Vector<double>> X, List<int> Y, int steps = 30, double learningRate = 0.3, int batchSize = 10, int maxsize = -1 )
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="X"></param>
        /// <param name="Y"></param>
        /// <param name="learningRate"></param>
        public void TrainBatch( List<Vector<double>> X, List<int> Y, double learningRate )
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x">ImageStream</param>
        /// <param name="y">Expected output</param>
        /// <returns></returns>
        public (List<Matrix<double>>, List<Vector<double>>) BackProp( Vector<double> x, int y )
        {
            throw new NotImplementedException();
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
            throw new NotImplementedException();
        }


    }
}
