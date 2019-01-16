using jlib;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TypTop
{
    class Program
    {
        private static Dictionary<string, double[]> TypEmbeddings;
        private static Dictionary<string, double[]> TopEmbeddings;
        private static DSSMModel TypModels;
        private static DSSMModel TopModels;
        static void Main(string[] args)
        {
            Console.WriteLine("Loading models...");
            TypModels = new DSSMModel("ACT Model", @"..\..\..\..\..\models\ACT.Prefix", @"..\..\..\..\..\models\ACT.Prefix", DSSMHelper.Vocabulary, 10, false);
            TopModels = new DSSMModel("Bing CDSSM Model", @"..\..\..\..\..\models\CONSKM_3LAYER_300_300_300_FILTER6_TITLE_source", @"..\..\..\..\..\models\CONSKM_3LAYER_300_300_300_FILTER6_TITLE_source", DSSMHelper.Vocabulary, 10, false);
            Console.WriteLine("Loading query embeddings...");
            LoadQueries();

            while (true)
            {
                Console.Write("Enter query: ");
                string q = Console.ReadLine().ToLower().Trim().Replace("+", "\\+").Replace("-", "\\-");
                if (q.Length > 0)
                {
                    Dictionary<string, double> typResults = new Dictionary<string, double>();
                    Dictionary<string, double> topResults = new Dictionary<string, double>();
                    double[] qTypEmbeddings = TypModels.GetSourceEmbeddings(q);
                    double[] qTopEmbeddings = TopModels.GetSourceEmbeddings(q);

                    foreach (KeyValuePair<string, double[]> pair in TypEmbeddings)
                    {
                        double sim = NNModelUtils.CosineSim(qTypEmbeddings, pair.Value);
                        typResults[pair.Key] = sim;
                    }

                    foreach (KeyValuePair<string, double[]> pair in TopEmbeddings)
                    {
                        double sim = NNModelUtils.CosineSim(qTopEmbeddings, pair.Value);
                        topResults[pair.Key] = sim;
                    }

                    List<KeyValuePair<string, double>> typResultsList = typResults.ToList();
                    List<KeyValuePair<string, double>> topResultsList = topResults.ToList();

                    typResultsList.Sort((firstPair, nextPair) =>
                    {
                        return nextPair.Value.CompareTo(firstPair.Value);
                    }
                    );

                    topResultsList.Sort((firstPair, nextPair) =>
                    {
                        return nextPair.Value.CompareTo(firstPair.Value);
                    }
                    );

                    Console.WriteLine("==Typical==");
                    for (int i = 0; i < Math.Min(10, typResultsList.Count); i++)
                    {
                        Console.WriteLine(typResultsList[i].Key + " (" + typResultsList[i].Value.ToString() + ")");
                    }
                    Console.WriteLine();
                    Console.WriteLine("==Topical==");
                    for (int i = 0; i < Math.Min(10, topResultsList.Count); i++)
                    {
                        Console.WriteLine(topResultsList[i].Key + " (" + topResultsList[i].Value.ToString() + ")");
                    }
                    Console.WriteLine();
                    Console.WriteLine();
                }
            }
        }
        private static void LoadQueries()
        {
            TypEmbeddings = new Dictionary<string, double[]>();
            TopEmbeddings = new Dictionary<string, double[]>();
            StreamReader queryReader = new StreamReader(@"..\..\..\..\..\data\candidates.tsv");

            try
            {
                //int count = 0;
                while (!queryReader.EndOfStream)// && ++count <= 100000)
                {
                    string query = queryReader.ReadLine().Split('\t')[0];
                    TypEmbeddings[query] = TypModels.GetSourceEmbeddings(query);
                    TopEmbeddings[query] = TopModels.GetSourceEmbeddings(query);
                }
            }
            finally
            {
                queryReader.Close();
            }
        }
    }
}
