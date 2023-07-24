using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_mnist.Resources.Models
{
    class DigitImage
    {
        public int Width { get; set; } // 28
        public int Height { get; set; } // 28

        public double[,] Pixels { get; set; } // all image pixels
        public double[] label { get; set; } // 0-9

        public int labelDigit;

        public DigitImage(int width, int height, byte[,] Pixels, byte labelDigit)
        {
            this.Width = width;
            this.Height = height;

            this.Pixels = new double[width, height];

            for (int i = 0; i < Pixels.GetLength(0); i++)
            {
                for (int j = 0; j < Pixels.GetLength(1); j++)
                {
                    this.Pixels[i, j] = Convert.ToDouble(Pixels[i, j]) / 255;
                }
            }

            switch (labelDigit)
            {
                case 0:
                    label = new double[10] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                    break;
                case 1:
                    label = new double[10] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
                    break;
                case 2:
                    label = new double[10] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
                    break;
                case 3:
                    label = new double[10] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
                    break;
                case 4:
                    label = new double[10] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
                    break;
                case 5:
                    label = new double[10] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
                    break;
                case 6:
                    label = new double[10] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
                    break;
                case 7:
                    label = new double[10] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
                    break;
                case 8:
                    label = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
                    break;
                case 9:
                    label = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 };
                    break;
            }

            this.labelDigit = labelDigit;
        }
    }
}
