(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
//#r "../../packages/formatting/FSharp.Plotly/lib/netstandard2.0/Fsharp.Plotly.dll"
#r "netstandard"
#r "../../lib/Formatting/FSharp.Plotly.dll"
open FSharp.Plotly
(**
Binary classification and PCA: Fraud detection in credit cards
==============================================================



| ML.NET version | API type          | Status                        | App Type    | Data type | Scenario            | ML Task                   | Algorithms                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v0.10           | Dynamic API | Up-to-date | Two console apps | .csv file | Fraud Detection | Two-class classification | FastTree Binary Classification |

In this introductory sample, you'll see how to use FSharpML on top of [ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) to predict a credit card fraud. In the world of machine learning, this type of prediction is known as binary classification.

//## API version: Dynamic and Estimators-based API
//It is important to note that this sample uses the dynamic API with Estimators.

Problem
-------

This problem is centered around predicting if credit card transaction (with its related info/variables) is a fraud or no. 

The input information of the transactions contain only numerical input variables which are the result of PCA transformations. Unfortunately, due to confidentiality issues, the original features and additional background information are not available, but the way you build the model doesn't change.  

Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 

The feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

Using those datasets you build a model that when predicting it will analyze a transaction's input variables and predict a fraud value of false or true.

DataSet
-------

The training and testing data is based on a public [dataset available at Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) originally from Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles), collected and analysed during a research collaboration. 

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.

By: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence and http://mlg.ulb.ac.be/ARTML

## ML Task - [Binary Classification](https://en.wikipedia.org/wiki/Binary_classification)

Binary or binomial classification is the task of classifying the elements of a given set into two groups (predicting which group each one belongs to) on the basis of a classification rule. Contexts requiring a decision as to whether or not an item has some qualitative property, some specified characteristic
  
Solution
--------

To solve this problem, first you need to build a machine learning model. Then you train the model on existing training data, evaluate how good its accuracy is, and lastly you consume the model (deploying the built model in a different app) to predict a fraud for a sample credit card transaction.

![Build -> Train -> Evaluate -> Consume](../files/img/modelpipeline.png)


1. Build and train the model
----------------------------

FSharpML containing two complementary parts named EstimatorModel and TransformerModel covering the full machine lerarning workflow. In order to build an ML model and fit it to the training data we use EstimatorModel.
The 'fit' function in EstimatorModel applied on training data results into the TransformerModel that represents the trained model able to transform other data of the same shape and is used int the second part to evaluate and consume the model.

Building a model includes:

- Define the data's schema maped to the datasets to read with a DataReader

- Split data for training and tests

- Create an Estimator and transform the data with a ConcatEstimator() and Normalize by Mean Variance. 

- Choosing a trainer/learning algorithm (FastTree) to train the model with.

**)

#load "../../FSharpML.fsx"


open System;
open Microsoft.ML
open Microsoft.ML.Data;
open FSharpML
open FSharpML.EstimatorModel
open FSharpML.TransformerModel
open System.IO
open System.IO.Compression


(*** hide ***)
let extractZip inPath outPath = 
    let readFromZip path = 
        seq { let sr = 
                new StreamReader(
                  new GZipStream(
                    File.OpenRead(path), 
                    CompressionMode.Decompress))
              while not sr.EndOfStream do
                yield sr.ReadLine()
              }

    let write (path:string) (data:seq<string>) =
        let sw = new StreamWriter(path)
        data
        |> Seq.iter (fun line -> sw.WriteLine line)
    write outPath (readFromZip inPath)



/// Data models used as an input to prediction function.
[<CLIMutable>]
type TransactionObservation = {
    [<ColumnName("Class")>]
    Label: int
    V1 : float32
    V2 : float32
    V3 : float32
    V4 : float32
    V5 : float32
    V6 : float32
    V7 : float32
    V8 : float32
    V9 : float32
    V10: float32
    V11: float32
    V12: float32
    V13: float32
    V14: float32
    V15: float32
    V16: float32
    V17: float32
    V18: float32
    V19: float32
    V20: float32
    V21: float32
    V22: float32
    V23: float32
    V24: float32
    V25: float32
    V26: float32
    V27: float32
    V28: float32
    Amount: float32
    }

[<CLIMutable>]
type TransactionFraudPrediction = {
    Label: bool
    PredictedLabel: bool
    Score: float32
    Probability: float32
    }


//let inputFile = Path.Combine (dataDirectory, "creditcard.csv")
//let trainFile = Path.Combine (dataDirectory, "trainData.csv")
//let testFile = Path.Combine (dataDirectory, "testData.csv")

//Create the MLContext to share across components for deterministic results
let mlContext = MLContext(seed = Nullable 1) // Seed set to any number so you
                                             // have a deterministic environment

// STEP 1: Common data loading configuration
let fullData =     
    //let inzip    = __SOURCE_DIRECTORY__  + "./data/creditcard.csv.gz"
    //let dataFile = System.IO.Path.GetTempPath() + "creditcard.csv"
    //extractZip inzip dataFile 
    let dataFile = System.IO.Path.GetTempPath() + "creditcard.csv"
    dataFile 
    |> DataModel.fromTextFileWith<TransactionObservation> mlContext ',' true 
    
let t = Data.getColumn<int> mlContext "Class" fullData.Dataview 


//let trainingData, testingData = 
//    fullData
//    |> DataModel.BinaryClassification.trainTestSplit 0.2 

let featureColumnNames = 
    fullData.Dataview.Schema    
    |> Seq.map (fun column -> column.Name)
    |> Seq.filter (fun name -> name <> "Class")
    |> Seq.filter (fun name -> name <> "Label")
    |> Seq.filter (fun name -> name <> "StratificationColumn")
    |> Seq.toArray

////STEP 2: Process data, create and train the model 
//let model = 
//    EstimatorModel.create mlContext
//    // Process data transformations in pipeline
//    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Conversion.ValueMap([0;1],[false;true],(struct (DefaultColumnNames.Label,"Class")))) // ConvertType(DefaultColumnNames.Label,"Class",DataKind.Bool))
//    |> EstimatorModel.Transforms.concatenate DefaultColumnNames.Features featureColumnNames
//    |> EstimatorModel.Transforms.normalizeMeanVariance "FeaturesNormalizedByMeanVar" DefaultColumnNames.Features  
    
//    // Create the model
//    |> EstimatorModel.appendBy (fun mlc -> 
//        mlc.BinaryClassification.Trainers.FastTree
//            (
//                DefaultColumnNames.Label, 
//                DefaultColumnNames.Features, 
//                numLeaves = 20, 
//                numTrees = 100, 
//                minDatapointsInLeaves = 10, 
//                learningRate = 0.2
//            ) )
 
//    // Train the model
//    |> EstimatorModel.fit fullData.Dataview //trainingData.Dataview

(**
2. Evaluate and consume the model
---------------------------------

TransformerModel is used to evaluate the model and make prediction on independant data.

**)

//// STEP3: Run the prediciton on the test data
//let predictions =
//    model
//    |> TransformerModel.transform trainingData.Dataview


//// STEP4: Evaluate accuracy of the model
//let metrics = 
//    model
//    |> Evaluation.BinaryClassification.evaluate trainingData.Dataview



//let predict = 
//    TransformerModel.createPredictionEngine<_,TransactionObservation,TransactionFraudPrediction> model


////let toSeq<'TRow when 'TRow :not struct and 'TRow : (new: unit -> 'TRow) > (dataModel:DataModel.DataModel<'a :> obj>) =
////    dataModel.Context.CreateEnumerable<'TRow>(dataModel.Dataview, reuseRowObject = false)

//DataModel.toSeq<TransactionObservation> testingData
//|> Seq.filter (fun x -> x.Label = true)
//// use 5 observations from the test data
//|> Seq.take 5
//|> Seq.iter (fun testData -> 
//    let prediction = predict testData
//    printfn "%A" prediction
//    printfn "------"
//    )

//let addColumn =
    //// Compute row count
    //Microsoft.ML.Data.DataViewUtils.ComputeRowCount
    // Get single values
    //Microsoft.ML.Data.ColumnCursorExtensions
    
    //Microsoft.ML.Data.LambdaFilter

    // Stacks two dataciews
    //Microsoft.ML.Data.AppendRowsDataView

    //Microsoft.ML.Data.CursoringUtils.CreateEnumerable


    //Microsoft.ML.Data.RowCursorUtils
    
        





//let advb = Microsoft.ML.Data.ArrayDataViewBuilder(mlContext)
//advb.AddColumn<bool>("Label",bt,[|ls|])
//advb.AddColumn<ReadOnlyMemory<char>>("Text",tt,[|ts|])
//let dataView = advb.GetDataView()


