using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace AI_mnist.Resources.Models
{
    internal class WeightLoader
    {
        public double[,] Load(string path)
        {
            var formatter = new BinaryFormatter();
            var inputStream = File.OpenRead(path);
            var result = (double[,])formatter.Deserialize(inputStream);
            inputStream.Close();

            return result;
        }
    }
}
