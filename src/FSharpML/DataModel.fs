namespace FSharpML

open System
open Microsoft.ML
open Microsoft.Data.DataView




module DataModel =
    open System.Data
    open Data

    type DataModel<'info> = {
        Context  : MLContext
        Dataview : IDataView        
        Metadata : 'info option
        }
    
    /// Creates a data model record (use createDataModelWith to include metadata information)
    let createDataModel context dataview =
        {Context=context;Dataview=dataview;Metadata=None}

    /// Creates a data model record with metadata information
    let createDataModelWith mlc dataview metaData  =
        {Context=mlc;Dataview=dataview;Metadata=Some metaData}

    /// Creates a data model from a given data view
    let ofDataview mlc dataview = 
        {Context=mlc;Dataview=dataview;Metadata=None}

    /// Creates a data model from seq<'Trow> 
    let ofSeq mlc (data:seq<'Trow>) =
        let dv = Data.readFromEnumerable mlc data
        {Context=mlc;Dataview=dv;Metadata=None}
 
    let toSeq<'a,  'TRow when 'TRow :not struct and 'TRow : (new: unit -> 'TRow) > (dataModel:DataModel<'a>) =
        dataModel.Context.CreateEnumerable<'TRow>(dataModel.Dataview, reuseRowObject = false)
    
    /// Reads a data model from a text file 
    let fromTextFile<'Trow> mlc path = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.readFromTextFile mlc '\t' true columns path
        {Context=mlc;Dataview=dv;Metadata=None}

    /// Reads a data model from a text file
    let fromTextFileWith<'Trow> mlc separatorChar hasHeader  path = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.readFromTextFile mlc separatorChar hasHeader columns path
        {Context=mlc;Dataview=dv;Metadata=None}
    
    /// Reads a data model from a text file
    let fromTextStreamWith<'Trow> mlc separatorChar hasHeader  stream = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.readFromTextStream mlc separatorChar hasHeader columns stream
        {Context=mlc;Dataview=dv;Metadata=None}

    /// Reads a data model from a text stream 
    let fromTextStream<'Trow> mlc stream =         
        fromTextStreamWith<'Trow> mlc '\t' true stream

    /// Reads a data view from text file as stream input
    let readFromTextStream (mlc:MLContext) separatorChar hasHeader columns (stream:#IO.Stream) = 
        let txtld =
            new TextLoader(
                mlc,
                columns = columns,
                hasHeader = hasHeader,
                separatorChar = separatorChar)
        
        txtld.Read(stream)


    /// Reads a data model from a binary file 
    let fromBinaryStream<'Trow> mlc stream = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.readFromBinary mlc stream
        {Context=mlc;Dataview=dv;Metadata=None}

    /// Reads a data model from a binary file
    let fromBinaryStreamWith<'Trow> mlc separatorChar hasHeader stream = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.readFromTextFile mlc separatorChar hasHeader columns stream
        {Context=mlc;Dataview=dv;Metadata=None}


    /// Returns the MLcontext
    let getContext (dataModel:DataModel<_>) = dataModel.Context

    /// Returns the data view
    let toDataview (dataModel:DataModel<_>) = dataModel.Dataview

    /// Try to return meta data else None
    let tryGetMetadata (dataModel:DataModel<_>) = 
        dataModel.Metadata

    
    /// Saves data view to text file
    let saveAsText separatorChar hasHeader (stream:System.IO.Stream) (dataModel:DataModel<_>) = 
        dataModel.Context.Data.SaveAsText(dataModel.Dataview, stream, separatorChar, hasHeader)

    /// Saves data view to binary file
    let saveAsBinary (stream:System.IO.Stream) (dataModel:DataModel<_>) = 
        dataModel.Context.Data.SaveAsBinary(dataModel.Dataview, stream)

    /// Returns all values of a column
    let getColumn<'a> columnName (dataModel:DataModel<_>) = 
        dataModel.Dataview.GetColumn<'a>(dataModel.Context,columnName)
    
    /// Keeps only those rows that are between lower and upper range condition
    let appendFilterByColumn columnName lower upper (dataModel:DataModel<_>) = 
        let df = dataModel.Context.Data.FilterByColumn(dataModel.Dataview,columnName,lower,upper)
        {dataModel with Dataview=df}



    /// Metadata info for a traintestsplit    
    type TrainTestSplitInfo = {
        SplitFraction       : float
        StratificationColumn: string option
        }
    
    // Creates metadata info for a traintestsplit    
    let createTrainTestSplitInfo splitFraction stratificationColumn =
        {SplitFraction=splitFraction; StratificationColumn=stratificationColumn}
 
    
    type BinaryClassification =
        
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testfraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let train,test = (dataModel.Context.BinaryClassification.TrainTestSplit(dataModel.Dataview,testfraction,stratification,seed)).ToTuple() 
                    let trainInfo  = createTrainTestSplitInfo  (1. - testfraction) Stratification
                    let testInfo   = createTrainTestSplitInfo testfraction Stratification
                    (
                        createDataModelWith dataModel.Context train trainInfo,
                        createDataModelWith dataModel.Context test testInfo                    
                    )
    
    module BinaryClassification =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit testfraction (dataModel:DataModel<_>) =
            BinaryClassification.trainTestSplitWith(Testfraction=testfraction) dataModel


    type MulticlassClassification =
        
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (                
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let struct (train,test) = dataModel.Context.MulticlassClassification.TrainTestSplit(dataModel.Dataview,testFraction,stratification,seed)
                    let trainInfo  = createTrainTestSplitInfo testFraction Stratification
                    let testInfo   = createTrainTestSplitInfo (1. - testFraction) Stratification
                    (
                        createDataModelWith dataModel.Context train trainInfo,
                        createDataModelWith dataModel.Context test testInfo                    
                    )
    
    module MulticlassClassification =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit testfraction (dataModel:DataModel<_>) =
            BinaryClassification.trainTestSplitWith(Testfraction=testfraction) dataModel
                    

    type Regression =
        
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (                
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let struct (train,test) = dataModel.Context.Regression.TrainTestSplit(dataModel.Dataview,testFraction,stratification,seed)                    
                    let trainInfo  = createTrainTestSplitInfo testFraction Stratification
                    let testInfo   = createTrainTestSplitInfo (1. - testFraction) Stratification
                    (
                        createDataModelWith dataModel.Context train trainInfo,
                        createDataModelWith dataModel.Context test testInfo                     
                    )
    
    module Regression =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit testfraction (dataModel:DataModel<_>) =
            Regression.trainTestSplitWith(Testfraction=testfraction) dataModel                    

    type Clustering =
    
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (                
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let struct (train,test) = dataModel.Context.Clustering.TrainTestSplit(dataModel.Dataview,testFraction,stratification,seed)                    
                    let trainInfo  = createTrainTestSplitInfo testFraction Stratification
                    let testInfo   = createTrainTestSplitInfo (1. - testFraction) Stratification
                    (
                        createDataModelWith dataModel.Context train trainInfo,
                        createDataModelWith dataModel.Context test testInfo                     
                    )
                    
    
    module Clustering =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit testfraction (dataModel:DataModel<_>) =
            Clustering.trainTestSplitWith(Testfraction=testfraction) dataModel   
                    
                       
    type Ranking =
        
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (                
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let struct (train,test) = dataModel.Context.Ranking.TrainTestSplit(dataModel.Dataview,testFraction,stratification,seed)                    
                    let trainInfo  = createTrainTestSplitInfo testFraction Stratification
                    let testInfo   = createTrainTestSplitInfo (1. - testFraction) Stratification
                    (
                        createDataModelWith dataModel.Context train trainInfo,
                        createDataModelWith dataModel.Context test testInfo                
                    )

     module Ranking =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit testfraction (dataModel:DataModel<_>) =
            Ranking.trainTestSplitWith(Testfraction=testfraction) dataModel          

        
        //Not implemented in ML.net: AnomalyDetection