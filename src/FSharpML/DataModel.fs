namespace FSharpML

open System
open Microsoft.ML
open Microsoft.ML.Data




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
    let ofDataview<'info> mlc dataview = 
        {Context=mlc;Dataview=dataview;Metadata=None}

    /// Creates a data model from seq<'Trow> 
    let ofSeq mlc (data:seq<'Trow>) =
        let dv = Data.loadFromEnumerable mlc data
        {Context=mlc;Dataview=dv;Metadata=None}
 
    let toSeq<'a,  'TRow when 'TRow :not struct and 'TRow : (new: unit -> 'TRow) > (dataModel:DataModel<'a>) =
        dataModel.Context.Data.CreateEnumerable<'TRow>(dataModel.Dataview, reuseRowObject = false)
    
    /// Reads a data model from a text file 
    let fromTextFile<'Trow> mlc path = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.loadFromTextFile mlc '\t' true columns path
        {Context=mlc;Dataview=dv;Metadata=None}

    /// Reads a data model from a text file
    let fromTextFileWith<'Trow> mlc separatorChar hasHeader  path = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.loadFromTextFile mlc separatorChar hasHeader columns path
        {Context=mlc;Dataview=dv;Metadata=None}
    
    ///// Reads a data model from a text file
    //let fromTextStreamWith<'Trow> mlc separatorChar hasHeader  stream = 
    //    let columns = TextLoader.columnsFrom typeof<'Trow>
    //    let dv = Data.loadFromTextStream mlc separatorChar hasHeader columns stream
    //    {Context=mlc;Dataview=dv;Metadata=None}

    ///// Reads a data model from a text stream 
    //let fromTextStream<'Trow> mlc stream =         
    //    fromTextStreamWith<'Trow> mlc '\t' true stream

    ///// Reads a data view from text file as stream input
    //let readFromTextStream (mlc:MLContext) separatorChar hasHeader columns (stream:#IO.Stream) = 
    //    let txtld =
    //        new TextLoader(
    //            mlc,
    //            columns = columns,
    //            hasHeader = hasHeader,
    //            separatorChar = separatorChar)
        
    //    txtld.Read(stream)


    /// Reads a data model from a binary file 
    let fromBinaryStream<'Trow> mlc stream = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.loadFromBinary mlc stream
        {Context=mlc;Dataview=dv;Metadata=None}

    /// Reads a data model from a binary file
    let fromBinaryStreamWith<'Trow> mlc separatorChar hasHeader stream = 
        let columns = TextLoader.columnsFrom typeof<'Trow>
        let dv = Data.loadFromTextFile mlc separatorChar hasHeader columns stream
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
    let getColumn<'a> (columnName:string) (dataModel:DataModel<_>) = 
        dataModel.Dataview.GetColumn<'a>(columnName)
    
    /// Keeps only those rows that are between lower and upper range condition
    let appendFilterByColumn columnName lower upper (dataModel:DataModel<_>) = 
        let df = dataModel.Context.Data.FilterRowsByColumn(dataModel.Dataview,columnName,lower,upper)
        {dataModel with Dataview=df}



    /// Metadata info for a traintestsplit    
    type TrainTestSplitInfo = {
        SplitFraction       : float
        StratificationColumn: string option
        }
    
    // Creates metadata info for a traintestsplit    
    let createTrainTestSplitInfo splitFraction stratificationColumn =
        {SplitFraction=splitFraction; StratificationColumn=stratificationColumn}
 
    
    /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
    let trainTestSplitWith testfraction stratification seed (dataModel:DataModel<_>) = 
        let trainTestSplit = dataModel.Context.Data.TrainTestSplit(dataModel.Dataview,testfraction,stratification,Nullable seed)
        let trainingDataView = trainTestSplit.TrainSet
        let testDataView = trainTestSplit.TestSet        
        let trainInfo  = createTrainTestSplitInfo  (1. - testfraction) (Some stratification)
        let testInfo   = createTrainTestSplitInfo testfraction (Some stratification)
        (
            createDataModelWith dataModel.Context trainingDataView trainInfo,
            createDataModelWith dataModel.Context testDataView testInfo                    
        )


    /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
    let trainTestSplitWithStrat testfraction stratification (dataModel:DataModel<_>) = 
        let trainTestSplit = dataModel.Context.Data.TrainTestSplit(dataModel.Dataview,testfraction,stratification)
        let trainingDataView = trainTestSplit.TrainSet
        let testDataView = trainTestSplit.TestSet        
        let trainInfo  = createTrainTestSplitInfo  (1. - testfraction) (Some stratification)
        let testInfo   = createTrainTestSplitInfo testfraction (Some stratification)
        (
            createDataModelWith dataModel.Context trainingDataView trainInfo,
            createDataModelWith dataModel.Context testDataView testInfo                    
        )
    

    /// Splits a dataset into the train set and the test set according to the given fraction.
    let trainTestSplit testfraction (dataModel:DataModel<_>) =
        let trainTestSplit = dataModel.Context.Data.TrainTestSplit(dataModel.Dataview,testfraction)
        let trainingDataView = trainTestSplit.TrainSet
        let testDataView = trainTestSplit.TestSet        
        let trainInfo  = createTrainTestSplitInfo  (1. - testfraction) None
        let testInfo   = createTrainTestSplitInfo testfraction None
        (
            createDataModelWith dataModel.Context trainingDataView trainInfo,
            createDataModelWith dataModel.Context testDataView testInfo                    
        )  
    
    