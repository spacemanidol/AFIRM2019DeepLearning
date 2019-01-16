using jlib;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Analogies
{
    class Program
    {
        private static Dictionary<string, double[]> Queries;
        private static DSSMModel Models;
        static void Main(string[] args)
        {
            Console.WriteLine("Loading model...");
            Models = new DSSMModel("SearchTrails (20140608) Symmetric Model", @"..\..\..\..\..\models\SearchTrails.SIGIR2015.Bing.SessionPairs.Symmetric.Deep.20150106", @"..\..\..\..\..\models\SearchTrails.SIGIR2015.Bing.SessionPairs.Symmetric.Deep.20150106", DSSMHelper.Vocabulary, 10, false);
            Console.WriteLine("Loading query embeddings...");
            LoadQueries();

            while (true)
            {
                Console.Write("Enter query: ");
                string q = Console.ReadLine().ToLower().Trim().Replace("+", "\\+").Replace("-", "\\-");
                if (q.Length > 0)
                {
                    Dictionary<string, double> results = new Dictionary<string, double>();
                    double[] sourceEmbeddings = new double[Models.NumOutputNode];

                    string[] parts = q.Split(new char[] { '\\' }, StringSplitOptions.RemoveEmptyEntries);

                    foreach (string part in parts)
                    {
                        string subtext = part.Trim();
                        int sign = 1;

                        if (subtext[0] == '+')
                        {
                            subtext = subtext.Remove(0, 1).Trim();
                        }
                        else if (subtext[0] == '-')
                        {
                            subtext = subtext.Remove(0, 1).Trim();
                            sign = -1;
                        }

                        double[] qEmbeddings = Models.GetSourceEmbeddings(subtext);
                        double norm = Math.Max(1e-20, NNModelUtils.Norm(qEmbeddings));

                        for (int i = 0; i < qEmbeddings.Length; i++)
                        {
                            sourceEmbeddings[i] += (sign * qEmbeddings[i] / norm);
                        }
                    }

                    foreach (KeyValuePair<string, double[]> pair in Queries)
                    {
                        double sim = NNModelUtils.CosineSim(sourceEmbeddings, pair.Value);
                        results[pair.Key] = sim;
                    }

                    List<KeyValuePair<string, double>> resultsList = results.ToList();

                    resultsList.Sort((firstPair, nextPair) =>
                    {
                        return nextPair.Value.CompareTo(firstPair.Value);
                    }
                    );

                    for (int i = 0; i < Math.Min(10, resultsList.Count); i++)
                    {
                        Console.WriteLine(resultsList[i].Key + " (" + resultsList[i].Value.ToString() + ")");
                    }
                    Console.WriteLine();
                    Console.WriteLine();
                }
            }
        }
        private static void LoadQueries()
        {
            Queries = new Dictionary<string, double[]>();
            StreamReader queryReader = new StreamReader(@"..\..\..\..\..\data\candidates.tsv");

            try
            {
                //int count = 0;
                while (!queryReader.EndOfStream)// && ++count <= 100000)
                {
                    string query = queryReader.ReadLine().Split('\t')[0];
                    Queries[query] = Models.GetSourceEmbeddings(query);
                }
            }
            finally
            {
                queryReader.Close();
            }
        }
    }
}
