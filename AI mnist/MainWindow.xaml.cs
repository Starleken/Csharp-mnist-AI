using AI_mnist.Resources.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace AI_mnist
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private string trainImagesPath = "D:\\Projects\\MNIST\\train-images.idx3-ubyte";
        private string trainLabelsPath = "D:\\Projects\\MNIST\\train-labels.idx1-ubyte";

        private DigitImage[] digitImages;
        private int hiddenSize = 100;
        private double alpha = 0.000002;

        public MainWindow()
        {
            InitializeComponent();

            digitImages = LoadData(trainImagesPath, trainLabelsPath);
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {

            
            WeightLoader loader = new WeightLoader();

            double[,] weights0_1 = loader.Load("D:\\Weights\\Weights01.mnist");
            double[,] weights1_2 = loader.Load("D:\\Weights\\Weights12.mnist");

            byte[,] layer0Matrix = digitImages[1].Pixels;

            double[] layer0 = MatrixInVector(layer0Matrix);

            double[] layer1 = Dot(layer0, weights0_1);

            double[] layer2 = Dot(layer1, weights1_2);

            ResultText.Text = digitImages[615].labelDigit.ToString();

            foreach (double y in layer2)
            {
                ResultText.Text += $" {y}";
            }
        }

        private void Train()
        {
            WeightLoader loader = new WeightLoader();

            double[,] weights0_1 = loader.Load("D:\\Weights\\Weights01.mnist");
            double[,] weights1_2 = loader.Load("D:\\Weights\\Weights12.mnist");

            for (int i = 0; i < 20000; i++)
            {
                byte[,] layer0Matrix = digitImages[i].Pixels;

                double[] layer0 = MatrixInVector(layer0Matrix);

                double[] layer1 = Dot(layer0, weights0_1);

                double[] layer2 = Dot(layer1, weights1_2);

                double error = 0;

                for (int j = 0; j < layer2.Length; j++)
                {
                    error += digitImages[i].label[j] - layer2[j];
                }

                double[,] weightDelta_1_2 = new double[100, 10];

                for (int row = 0; row < weightDelta_1_2.GetLength(0); row++)
                {
                    for (int col = 0; col < weightDelta_1_2.GetLength(1); col++)
                    {
                        weightDelta_1_2[row, col] = layer1[row] * error;
                    }
                }

                for (int row = 0; row < weights1_2.GetLength(0); row++)
                {
                    for (int col = 0; col < weights1_2.GetLength(1); col++)
                    {
                        weights1_2[row, col] += weightDelta_1_2[row, col] * alpha;
                    }
                }


                if (i % 1000 == 0)
                {
                    ResultText.Text += $"{error}\n";
                }
            }

            WeightSaver weightSaver = new WeightSaver();
            weightSaver.Save(weights0_1, "D:\\Weights", "Weights01");
            weightSaver.Save(weights1_2, "D:\\Weights", "Weights12");
        }

        private DigitImage[] LoadData(string trainImagesPath, string trainLabelsPath)
        {
            int numImages = 60000;
            DigitImage[] result = new DigitImage[numImages];

            byte[,] pixels = new byte[28,28];

            FileStream images = new FileStream(trainImagesPath,FileMode.Open);
            FileStream labels = new FileStream(trainLabelsPath, FileMode.Open);

            BinaryReader imagesReader = new BinaryReader(images);
            BinaryReader labelsReader = new BinaryReader(labels);

            int magic1 = imagesReader.ReadInt32(); // stored as big endian
            magic1 = ReverseBytes(magic1); // convert to Intel format
            int imageCount = imagesReader.ReadInt32();
            imageCount = ReverseBytes(imageCount);
            int numRows = imagesReader.ReadInt32();
            numRows = ReverseBytes(numRows);
            int numCols = imagesReader.ReadInt32();
            numCols = ReverseBytes(numCols);
            int magic2 = labelsReader.ReadInt32();
            magic2 = ReverseBytes(magic2);
            int numLabels = labelsReader.ReadInt32();
            numLabels = ReverseBytes(numLabels);

            for (int i = 0; i < numImages; i++)
            {
                pixels = new byte[28, 28];
                for (int j = 0; j < 28; j++)
                {
                    for (int y = 0; y < 28; y++)
                    {
                        byte b = imagesReader.ReadByte();
                        pixels[j, y] = b;
                    }
                }
                byte label = labelsReader.ReadByte();
                DigitImage image = new DigitImage(28,28, pixels, label);
                result[i] = image;
            }

            imagesReader.Close();
            labelsReader.Close();
            images.Close();
            labels.Close();


            return result;
        }

        public static int ReverseBytes(int v)
        {
            byte[] intAsBytes = BitConverter.GetBytes(v);
            Array.Reverse(intAsBytes);
            return BitConverter.ToInt32(intAsBytes, 0);
        }

        public double[] MatrixInVector(byte[,] matrix)
        {
            double[] vector = new double[matrix.GetLength(0) * matrix.GetLength(1)];

            int current = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    vector[current] = matrix[i, j];
                    current++;
                }
            }

            return vector;
        }

        public double[] Dot(double[] vector, double[,] matrix)
        {
            if (vector.Length != matrix.GetLength(0))
            {
                throw new ArgumentException("Количество строк не равно количеству столбцов"); //TODO
            }

            double[] result = new double[matrix.GetLength(1)];

            for (int i = 0; i < matrix.GetLength(1); i++)
            {
                double number = 0;
                for (int j = 0; j < matrix.GetLength(0); j++)
                {
                    number += (vector[j] * matrix[j,i]);
                }
                result[i] = number;
            }

            return result;
        }

        public double NextDouble1(double MinValue, double MaxValue)
        {
            return new Random().NextDouble() * (MaxValue - MinValue) + MinValue;
        }
    }
}
