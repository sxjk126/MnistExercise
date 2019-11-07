using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace MnistExercise
{
    class Program
    {
        //filename for data set
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "handwritten_digits_large.csv");
        static void Main(string[] args)
        {
            //create a machine learning context
            var context = new MLContext();
            //load data
            Console.WriteLine("Loading data...");
            var dataView = context.Data.LoadFromTextFile(
                path: dataPath,
                columns: new[]
                {
                    new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
                    new TextLoader.Column("Number", DataKind.Single, 0)
                },
                hasHeader: false,
                separatorChar: ',');
            //split data into a training and test set
            var partitions = context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            //build a training pipeline
            //step 1: concatenate all feature columns
            var pipeline = context.Transforms.Concatenate(
                
                nameof(Digit.PixelValues))
                //cache data to speed up training
                .AppendCacheCheckpoint(context)
                //train the model with SDCA
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "Number",
                    featureColumnName: "Features"));

            //train the model
            Console.WriteLine("Training model...");
            var model = pipeline.Fit(partitions.TrainSet);

            //use the model to make prediction on the test data
            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(partitions.TestSet);
            //evaluate the predictions
            var metrics = context.MulticlassClassification.Evaluate(
                data: predictions,
                labelColumnName: "Number",
                scoreColumnName: "Score");

            //show evaluation metrics
            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:      {metrics.MicroAccuracy: 0.###}");
            Console.WriteLine($"    MacroAccuracy:      {metrics.MacroAccuracy: 0.###}");
            Console.WriteLine($"    LogLoss:            {metrics.LogLoss: #.###}");
            Console.WriteLine($"    LogLossReduction:   {metrics.LogLossReduction: #.###}");
            Console.WriteLine();

            //grab three digits from the data: 2, 7, and 9
            var digits = context.Data.CreateEnumerable<Digit>(dataView, reuseRowObject: false).ToString();
            var testDigits = new Digit[] { digits[5], digits[12], digits[20] };
            //create a prediction engine
            var engine = context.Model.CreatePredictionEngine<Digit, DigitPrediction>(model);
            // predict each test digit
            for (var i = 0; i < testDigits.Length; i++)
            {
                var prediction = engine.Predict(testDigits[i]);

                // show results
                Console.WriteLine($"Predicting test digit {i}...");
                for (var j = 0; j < 10; j++)
                {
                    Console.WriteLine($"  {j}: {prediction.Score[j]:P2}");
                }
                Console.WriteLine();
            }
        }
    }
    /// <summary>
    /// The Digit class represents one mnist digit.
    /// </summary>
    class Digit
    {
        [VectorType(785)] public float[] PixelValues;
    }
    /// <summary>
    /// The DigitPrediction class represents one digit prediction
    /// </summary>
    class DigitPrediction
    {
        public float[] Score;
    }
}
