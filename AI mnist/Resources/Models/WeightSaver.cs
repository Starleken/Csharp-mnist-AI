using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace AI_mnist.Resources.Models
{
    internal class WeightSaver
    {
        public void Save(double[,] matrix, string path, string fileName)
        {
            var formatter = new BinaryFormatter();
            FileStream outputStream = File.OpenWrite($"{path}/{fileName}.mnist");
            formatter.Serialize(outputStream, matrix);
            outputStream.Close();
        }
    }
}
