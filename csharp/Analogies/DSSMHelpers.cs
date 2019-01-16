using Common.Collections;
using Common.IO;
using Common.Text;
using Common.Utils;
using jlib;
using System;
using System.Collections.Generic;
using System.IO;

namespace Analogies
{
    public class DSSMHelper
    {
        private static Vocab vocabulary;
        public static Vocab Vocabulary
        {
            get { return vocabulary; }
        }

        static DSSMHelper()
        {
            vocabulary = new Vocab(false);
            vocabulary.Read(@"..\..\..\..\..\data\dssm-vocab.txt");
            vocabulary.Lock();
        }
    }

    public class DSSMModel
    {
        public string Name { get; set; }

        private Vocab Vocabulary;
        private DNN SourceModel;
        private DNN TargetModel;
        private int SourceWindowSize;
        private int TargetWindowSize;

        public int NumOutputNode { get { return SourceModel.NumOutputNode; } }

        public DSSMModel(string name, string commonModelPath, Vocab vocab, int windowSize, bool compatibilityMode)
            : this(name, commonModelPath, vocab, windowSize, windowSize, compatibilityMode)
        {
        }

        public DSSMModel(string name, string commonModelPath, Vocab vocab, int sourceWindowSize, int targetWindowSize, bool compatibilityMode)
        {
            this.Name = name;
            this.SourceModel = new DNN();
            this.SourceModel.LoadModelV0(commonModelPath, compatibilityMode);
            this.TargetModel = this.SourceModel;
            this.Vocabulary = vocab;
            this.SourceWindowSize = sourceWindowSize;
            this.TargetWindowSize = targetWindowSize;
        }

        public DSSMModel(string name, string sourceModelPath, string targetModelPath, Vocab vocab, int windowSize, bool compatibilityMode)
            : this(name, sourceModelPath, targetModelPath, vocab, windowSize, windowSize, compatibilityMode)
        {
        }

        public DSSMModel(string name, string sourceModelPath, string targetModelPath, Vocab vocab, int sourceWindowSize, int targetWindowSize, bool compatibilityMode)
        {
            this.Name = name;
            this.SourceModel = new DNN();
            this.TargetModel = new DNN();
            this.SourceModel.LoadModelV0(sourceModelPath, compatibilityMode);
            this.TargetModel.LoadModelV0(targetModelPath, compatibilityMode);
            this.Vocabulary = vocab;
            this.SourceWindowSize = sourceWindowSize;
            this.TargetWindowSize = targetWindowSize;
        }

        public double[] GetSourceEmbeddings(string text)
        {
            return GetEmbeddings(this.SourceModel, text, this.SourceWindowSize, FeatureType.l3g, 0);
        }

        public double[] GetTargetEmbeddings(string text)
        {
            return GetEmbeddings(this.TargetModel, text, this.TargetWindowSize, FeatureType.l3g, 0);
        }

        public double[] GetSourceEmbeddings(string text, FeatureType featureType, int nHashCount)
        {
            return GetEmbeddings(this.SourceModel, text, this.SourceWindowSize, featureType, nHashCount);
        }

        public double[] GetTargetEmbeddings(string text, FeatureType featureType, int nHashCount)
        {
            return GetEmbeddings(this.TargetModel, text, this.TargetWindowSize, featureType, nHashCount);
        }

        private double[] GetEmbeddings(DNN model, string text, int windowSize, FeatureType featureType, int nHashCount)
        {
            text = TextUtils.N1Normalize(text);

            if (text.Length == 0)
            {
                text = "#";
            }

            List<Dictionary<int, double>> rgSideWfs = new List<Dictionary<int, double>>();

            var featStrFeq = TextUtils.String2FeatStrSeq(text, 3, windowSize, featureType);  // letter N-gram

            if (featureType == FeatureType.wordhash)
            {
                rgSideWfs = TextUtils.StrFreq2IdFreq(featStrFeq, nHashCount);
            }
            else
            {
                rgSideWfs = TextUtils.StrFreq2IdFreq(featStrFeq, this.Vocabulary);
            }

            return model.Fprop(rgSideWfs);
        }
    }

    public class DNN
    {
        private Vocab m_Vocab = new Vocab(false);
        private List<double[][]> m_rgW = new List<double[][]>(); // project matrices
        private int m_CWinSize = 1;

        public int CWinSize { get { return m_CWinSize; } }

        public int GetWinSize(int iLayer)
        {
            if (iLayer == 0)
                return CWinSize;
            else
                return 1;
        }

        public int NumLayer { get { return m_rgW.Count; } }

        public int NumInputNode { get { return NumInputLayerNode(0); } }

        public int NumOutputNode { get { return NumOutputLayerNode(NumLayer - 1); } }

        public int NumInputLayerNode(int iLayer)
        {
            if (iLayer < 0 || iLayer >= NumLayer)
                return 0;
            return m_rgW[iLayer].Length / GetWinSize(iLayer);
        }

        public int NumOutputLayerNode(int iLayer)
        {
            if (iLayer < 0 || iLayer >= NumLayer)
                return 0;
            if (m_rgW[iLayer].Length == 0)
                return 0;
            return m_rgW[iLayer][0].Length;
        }


        /// <summary>
        /// load the neural net model of V0 format (Yelong's version)
        /// </summary>
        /// <param name="sFilename">model filename</param>
        public void LoadModelV0(string sModelFilename, bool loadModelInBackCompatibleMode)
        {
            List<int> rgn = new List<int>();

            BinaryReader br = new BinaryReader(File.Open(sModelFilename, FileMode.Open, FileAccess.Read));

            int nLayers = br.ReadInt32();
            rgn.Clear();
            for (int i = 0; i < nLayers; ++i)
                rgn.Add(br.ReadInt32());
            if (!loadModelInBackCompatibleMode)
            {
                // skip
                int mlink_num = br.ReadInt32();
                for (int i = 0; i < mlink_num; i++)
                {
                    int in_num = br.ReadInt32();
                    int out_num = br.ReadInt32();
                    float inithidbias = br.ReadSingle();
                    float initweightsigma = br.ReadSingle();
                    int win_size = br.ReadInt32();
                    int ntype = br.ReadInt32();
                    int pooltype = br.ReadInt32();
                    if (i == 0)
                    {
                        m_CWinSize = win_size;
                    }
                }
            }

            // init m_rgW
            for (int i = 0; i < nLayers - 1; ++i)
            {
                int ni = rgn[i];
                int no = rgn[i + 1];
                if (i == 0)
                {
                    ni = ni * m_CWinSize;
                }
                double[][] W = new double[ni][];
                for (int n = 0; n < ni; ++n)
                    W[n] = new double[no];
                m_rgW.Add(W);
            }
            if (loadModelInBackCompatibleMode)
            {
                // skip
                int mlink_num = br.ReadInt32();
                for (int i = 0; i < mlink_num; i++)
                {
                    int in_num = br.ReadInt32();
                    int out_num = br.ReadInt32();
                    float inithidbias = br.ReadSingle();
                    float initweightsigma = br.ReadSingle();
                }
            }
            // load weights
            for (int n = 0; n < nLayers - 1; ++n)
            {
                int nWeights = br.ReadInt32();
                int ni = NumInputLayerNode(n);
                int no = NumOutputLayerNode(n);
                if (n == 0)
                    ni = ni * m_CWinSize;
                if (nWeights != ni * no)
                    throw new Exception("Error in loading model weights");

                for (int i = 0; i < ni; ++i)
                    for (int o = 0; o < no; ++o)
                        m_rgW[n][i][o] = br.ReadSingle();

                int nBias = br.ReadInt32();
                if (nBias != no)
                    throw new Exception("Error in loading model bias");
                for (int o = 0; o < no; ++o)
                    br.ReadSingle();    // skip
            }

            br.Close();
        }

        /// <summary>
        /// forward propogation
        /// </summary>
        /// <param name="fvs"></param>
        /// <returns></returns>
        public double[] Fprop(List<Dictionary<int, double>> rgFvs)
        {
            if (NumLayer <= 0)
                throw new Exception("Error: the model is invalid");

            //if (fvs.Count != NumInputNode)
            //    throw new Exception("Error: the dim of input vector doesn't match the model in Fprop.");

            // convolutional layer
            List<double[]> rgY1 = new List<double[]>();
            for (int i = 0; i < rgFvs.Count; i++)
            {
                Dictionary<int, double> concat_fea = new Dictionary<int, double>();
                for (int ws = -CWinSize / 2; ws <= CWinSize / 2; ws++)
                {
                    if (i + ws >= 0 && i + ws < rgFvs.Count)
                    {
                        TextUtils.FeatureConcate(concat_fea, rgFvs[i + ws], (ws + CWinSize / 2) * NumInputNode);
                    }
                }
                rgY1.Add(NNModelUtils.ProjectionByATxSparse(m_rgW[0], concat_fea));
            }

            // max-pooling layer
            double[] Y = new double[NumOutputLayerNode(0)];
            for (int i = 0; i < NumOutputLayerNode(0); i++)
            {
                for (int k = 0; k < rgY1.Count; k++)
                {
                    if (k == 0 || rgY1[k][i] > Y[i])
                    {
                        Y[i] = rgY1[k][i];
                    }
                }
            }
            Y = NNModelUtils.Tanh(Y, 1.0);

            // semantic layers
            for (int n = 1; n < NumLayer; ++n)
            {
                Y = NNModelUtils.ProjectionByATx(m_rgW[n], Y);
                Y = NNModelUtils.Tanh(Y, 1.0);
            }

            return Y;
        }
    }

    public class Predictor
    {
        private DNN m_SrcModel = new DNN();
        private DNN m_TgtModel = new DNN();
        private Vocab m_V = new Vocab(false);
        private int m_LetterNgram = 3;
        bool LoadModelInBackCompatibleMode = false;
        public Predictor(string sSrcModel, string sTgtModel, bool loadModelInBackCompatibleMode)
        {
            LoadModelInBackCompatibleMode = loadModelInBackCompatibleMode;
            Load(sSrcModel, sTgtModel);
        }

        private void Load(string sSrcModel, string sTgtModel)
        {
            m_SrcModel.LoadModelV0(sSrcModel, LoadModelInBackCompatibleMode);
            m_TgtModel.LoadModelV0(sTgtModel, LoadModelInBackCompatibleMode);
        }

        public Predictor(string sSrcModel, string sTgtModel, string sVocab, int ngram)
        {
            Load(sSrcModel, sTgtModel, sVocab, ngram);
        }

        private void Load(string sSrcModel, string sTgtModel, string sVocab, int ngram)
        {
            m_SrcModel.LoadModelV0(sSrcModel, LoadModelInBackCompatibleMode);
            m_TgtModel.LoadModelV0(sTgtModel, LoadModelInBackCompatibleMode);

            m_V = new Vocab(false);
            using (StreamReader sr = new StreamReader(sVocab))
            {
                string sLine = "";
                while ((sLine = sr.ReadLine()) != null)
                {
                    string[] rgs = sLine.Trim().Split('\t');
                    m_V.Encode(rgs[0]);
                }
            }
            m_V.Lock();
            m_LetterNgram = ngram;
        }

        /// <summary>
        /// cosine sim btw src and tgt, where src/tgt are in matrix format
        /// </summary>
        /// <param name="src"></param>
        /// <param name="tgt"></param>
        /// <returns></returns>
        public double CosineSim(string src, string tgt)
        {
            double sim = 0;
            // Dictionary<int, double> srcVec = TextUtils.String2L3g(src, m_V, m_LetterNgram);
            // Dictionary<int, double> tgtVec = TextUtils.String2L3g(tgt, m_V, m_LetterNgram);
            List<Dictionary<int, double>> srcMt = TextUtils.String2Matrix(src);
            List<Dictionary<int, double>> tgtMt = TextUtils.String2Matrix(tgt);
            sim = NNModelUtils.CosineSim(m_SrcModel.Fprop(srcMt), m_TgtModel.Fprop(tgtMt));
            return sim;
        }

        /// <summary>
        /// compute sim btw src and tgt
        /// </summary>
        /// <param name="inTSV">input labeled data file</param>
        /// <param name="inSrc">in vector format</param>
        /// <param name="inTgt">in vector format</param>
        /// <param name="FeatName">feature name</param>
        /// <param name="outTSV">output score file</param>
        /// <param name="bOutputVector">whether to output vector</param>
        public void PredictingV1(string inTSV, string inSrc, string inTgt, string FeatName, string outTSV, bool bOutputVector)
        {
            StreamWriter sw = new StreamWriter(outTSV);
            StreamReader sr = null;
            if (inTSV != "")
                sr = new StreamReader(inTSV);

            Console.WriteLine("computing sim...");
            string sLine = "";
            int n = 0;

            if (sr != null)
            {
                sLine = sr.ReadLine();

                sw.Write("{0}\t{1}", sLine, FeatName);
                if (bOutputVector)
                {
                    for (int i = 0; i < m_SrcModel.NumOutputNode; ++i)
                        sw.Write("\t{0}_s{1}", FeatName, i);
                    for (int i = 0; i < m_TgtModel.NumOutputNode; ++i)
                        sw.Write("\t{0}_t{1}", FeatName, i);
                }
                sw.Write("\n");
            }

            sLine = "";
            foreach (Pair<string, string> p in PairEnum<string, string>.E(FileEnum.GetLines(inSrc), FileEnum.GetLines(inTgt)))
            {
                if (sr != null)
                    sLine = sr.ReadLine();

                List<Dictionary<int, double>> srcMt = TextUtils.String2Matrix(p.First);
                List<Dictionary<int, double>> tgtMt = TextUtils.String2Matrix(p.Second);
                double[] srcVt = m_SrcModel.Fprop(srcMt);
                double[] tgtVt = m_TgtModel.Fprop(tgtMt);
                double sim = NNModelUtils.CosineSim(srcVt, tgtVt);

                if (sr != null)
                    sw.Write("{0}\t{1}", sLine, (float)sim);
                else
                    sw.Write((float)sim);

                if (bOutputVector)
                {
                    for (int i = 0; i < m_SrcModel.NumOutputNode; ++i)
                        sw.Write("\t{0}", (float)srcVt[i]);
                    for (int i = 0; i < m_TgtModel.NumOutputNode; ++i)
                        sw.Write("\t{0}", (float)tgtVt[i]);
                }
                sw.Write("\n");

                n++; if (n % 1000 == 0) Console.Error.Write("{0}\r", n);
            }
            Console.WriteLine("{0} pairs.", n);

            sw.Close();
            if (sr != null)
                sr.Close();
        }

        /// <summary>
        /// compute sim btw src and tgt
        /// </summary>
        /// <param name="inSrc">in matrix format</param>
        /// <param name="inTgt">in matrix format</param>
        /// <param name="outScore"></param>
        public void PredictingV0(string inSrc, string inTgt, string outScore)
        {
            StreamWriter sw = new StreamWriter(outScore);

            int n = 0;
            foreach (Pair<string, string> p in
                PairEnum<string, string>.E(FileEnum.GetLines(inSrc), FileEnum.GetLines(inTgt)))
            {
                double sim = 0;

                if (p.First.Trim() != "" && p.Second.Trim() != "")
                    sim = CosineSim(p.First.Trim(), p.Second.Trim());

                // UInt32 score = (UInt32)((1 + sim) / 2.0 * Int32.MaxValue);
                sw.WriteLine(sim);
                n++;
                if (n % 1000 == 0)
                    Console.Write("{0}\r", n);
            }

            sw.Close();
        }
    }
}