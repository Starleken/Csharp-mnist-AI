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
using System.Windows.Forms;
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
        private double alpha = 0.05;

        public MainWindow()
        {
            InitializeComponent();

            digitImages = LoadData(trainImagesPath, trainLabelsPath);
        }

        private double Sigmoid(double output)
        {
            return 1 / (1 + Math.Exp(-output));
        }

        private double Sigmoid2Deriv(double output)
        {
            return output * (1 - output);
        }

        private double[] SoftMax(double[] x)
        {
            var x_exp = x.Select(Math.Exp);

            var sum_z_exp = x_exp.Sum();

            var softmax = x_exp.Select(i => i / sum_z_exp).ToArray<Double>();

            return softmax;
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            GetPrediction();
        }

        private void GetPrediction()
        {
            WeightLoader loader = new WeightLoader();

            double[,] weights0_1 = loader.Load("D:\\Weights\\Weights01.mnist");
            double[,] weights1_2 = loader.Load("D:\\Weights\\Weights12.mnist");

            OpenFileDialog openFileDlg = new OpenFileDialog();
            var dialogResult = openFileDlg.ShowDialog();

            string path = "";
            if (dialogResult == System.Windows.Forms.DialogResult.OK)
            {
                path = openFileDlg.FileName;
            }
            else
            {
                return;
            }

            byte[,] pixels = LoadImage(path);
            DigitImage image = new DigitImage(28, 28, pixels, 1);

            double[] result = image.label;
            double[,] layer0Matrix = image.Pixels;

            double[] layer0 = MatrixInVector(layer0Matrix);

            double[] layer1 = Dot(layer0, weights0_1);

            for (int j = 0; j < layer1.Length; j++)
            {
                layer1[j] = Sigmoid(layer1[j]);
            }

            double[] layer2 = Dot(layer1, weights1_2);

            ResultText.Text = GetMaxIndex(layer2).ToString();
            
        }

        private async void TrainAsync()
        {
            await Task.Run(() => Train());
        }

        private void Train()
        {
            double[,] weights0_1 = new double[784, hiddenSize];
            double[,] weights1_2 = new double[hiddenSize, 10];

            //Рандомизация весов
            for (int row = 0; row < weights0_1.GetLength(0); row++)
            {
                for (int col = 0; col < weights0_1.GetLength(1); col++)
                {
                    weights0_1[row, col] = 0.2 * new Random().NextDouble() - 0.1;
                }
            }

            for (int row = 0; row < weights1_2.GetLength(0); row++)
            {
                for (int col = 0; col < weights1_2.GetLength(1); col++)
                {
                    weights1_2[row, col] = 0.2 * new Random().NextDouble() - 0.1;
                }
            }

            //Начало итераций
            for (int iteration = 0; iteration < 20; iteration++)
            {
                double error = 0;
                float correct = 0;
                for (int i = 0; i < 2000; i++)
                {

                    //Подсчёт значений
                    double[] result = digitImages[i].label;
                    double[,] layer0Matrix = digitImages[i].Pixels;

                    double[] layer0 = MatrixInVector(layer0Matrix);

                    double[] layer1 = Dot(layer0, weights0_1);

                    for (int j = 0; j < layer1.Length; j++)
                    {
                        layer1[j] = Sigmoid(layer1[j]);
                    }

                    double[] layer2 = Dot(layer1, weights1_2);

                    //Расчёт ошибки
                    for (int j = 0; j < layer2.Length; j++)
                    {
                        error += result[j] - layer2[j];
                    }
                    if (GetMaxIndex(layer2) == GetMaxIndex(result))
                    {
                        correct += 1;
                    }
                    this.Dispatcher.Invoke(() =>
                    {
                        IterationTextBlock.Text = $"Iteration: {iteration}, image: {i}";
                    });

                    //Вычисление delta
                    double[] layer2_delta = new double[layer2.Length];
                    double[] layer1_delta = new double[layer1.Length];

                    for (int j = 0; j < layer2_delta.Length; j++)
                    {
                        layer2_delta[j] = result[j] - layer2[j];
                    }


                    layer1_delta = Dot(layer2_delta, Transpose(weights1_2));

                    for (int j = 0; j < layer1_delta.Length; j++)
                    {
                        layer1_delta[j] *= Sigmoid2Deriv(layer1[j]);
                    }

                    //Вычисление весов для добавления
                    double[,] weightDelta_0_1 = Dot2Vectors(layer0, layer1_delta);
                    double[,] weightDelta_1_2 = Dot2Vectors(layer1, layer2_delta);

                    //Добавление весов
                    for (int row = 0; row < weights1_2.GetLength(0); row++)
                    {
                        for (int col = 0; col < weights1_2.GetLength(1); col++)
                        {
                            weights1_2[row, col] += weightDelta_1_2[row, col] * alpha;
                        }
                    }

                    for (int row = 0; row < weights0_1.GetLength(0); row++)
                    {
                        for (int col = 0; col < weights0_1.GetLength(1); col++)
                        {
                            weights0_1[row, col] += weightDelta_0_1[row, col] * alpha;
                        }
                    }
                }
                this.Dispatcher.Invoke(() =>
                {
                    CorrectionTextBlock.Text = $"{error / 1000}     {correct / 1000}";
                });
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

        public int GetMaxIndex(double[] vector)
        {
            double number = Int32.MinValue;
            int index = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                if (number < vector[i])
                {
                    number = vector[i];
                    index = i;
                }
            }

            return index;
        }

        public double[] MatrixInVector(double[,] matrix)
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

        public double[,] Transpose(double[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);
            double[,] result = new double[h, w];
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }
            return result;
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

        public double[,] Dot2Vectors(double[] firstVector, double[] secondVector)
        {
            double[,] matrix = new double[firstVector.Length, secondVector.Length];

            for (int col = 0; col < secondVector.Length; col++)
            {
                for (int row = 0; row < firstVector.Length; row++)
                {
                    matrix[row, col] = firstVector[row] * secondVector[col];
                }
            }

            return matrix;
        }

        public byte[,] LoadImage(string path)
        {
            Uri myUri = new Uri(path, UriKind.RelativeOrAbsolute);
            BmpBitmapDecoder decoder = new BmpBitmapDecoder(myUri, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.Default);
            BitmapSource bs = decoder.Frames[0];
            //Конвертируем изображение в оттенки серого
            FormatConvertedBitmap fcb = new FormatConvertedBitmap(bs, PixelFormats.Gray8, BitmapPalettes.BlackAndWhite, 1);
            bs = fcb;
            byte[] arr = new byte[(int)(bs.Width * bs.Height)];
            //Извлекаем пиксели
            bs.CopyPixels(arr, (int)(8 * bs.Width) / 8, 0);
            int count = 0;
            byte[,] img = new byte[(int)bs.Height, (int)bs.Width];
            //формируем двумерный массив
            for (int i = 0; i < bs.Height; ++i)
            {
                for (int j = 0; j < bs.Width; ++j)
                {
                    img[i, j] = arr[count++];
                }
            }

            DigitImage.Source = bs;

            return img;
        }

        private void TrainButton_Click(object sender, RoutedEventArgs e)
        {
            TrainAsync();
        }
    }
}
