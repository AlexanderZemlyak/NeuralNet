{$reference 'Compiler.dll'}
{$reference 'Errors.dll'}
{$reference 'TreeConverter.dll'}
{$reference 'SyntaxTree.dll'}
{$reference 'SemanticTree.dll'}
{$reference 'SyntaxVisitors.dll'}
{$reference 'NetGenerator.dll'}
{$reference 'ParserTools.dll'}
{$reference 'LanguageIntegrator.dll'}
{$reference 'JSON_DLL\Newtonsoft.Json.dll'}
//{$reference CompilerTools.dll}
//{$reference Localization.dll}
//{$reference System.Windows.Forms.dll}
//{$reference LanguageIntegrator.dll}

uses PascalABCCompiler, Newtonsoft, System.IO;

begin
  
  Languages.Integration.LanguageIntegrator.LoadAllLanguages();
  
  var comp := new Compiler();
  
  var filePath := 'D:\DesktopFiles\Дипломная\NeuralNetForPascal\train_data.json';
  var jsonFromFile := &File.ReadAllText(filePath);
  
  var samples := Newtonsoft.Json.JsonConvert.DeserializeObject&<List<Dictionary<string, string>>>(jsonFromFile);
  
  var filesCount := 5277;
  
  for var i := 0 to filesCount - 1 do
  begin

    // var oldOutput := samples[i]['output'];
    
    var fileName := $'{i}.pas';
    // WriteAllText(fileName, oldOutput);
    
    var co: CompilerOptions := new CompilerOptions(Path.GetFullPath(fileName), CompilerOptions.OutputType.ConsoleApplicaton);
    co.Debug := false;
    co.SearchDirectories := ['D:\DesktopFiles\Дипломная\PABCNet source\pascalabcnet\bin\Lib'].ToList();
    co.UseDllForSystemUnits := false;
    co.RunWithEnvironment := false; 
    comp.InternalDebug.CodeGeneration := false;
    comp.ErrorsList.Clear();
    comp.Warnings.Clear();
    
    comp.Compile(co);
    
    if comp.ErrorsList.Count > 0 then
    begin
      var err := comp.ErrorsList.Last();
      
      samples[i]['compilationFailed'] := 'true';
      
      Println($'Ошибка при компиляции файла {fileName}:{NewLine}{err}{NewLine}');
    end
    else
    begin
      samples[i]['output'] := comp.Warnings[0].Message;
    end;
    
  end;
  
  var updatedJson := Newtonsoft.Json.JsonConvert.SerializeObject(samples, Newtonsoft.Json.Formatting.Indented);
  
  var newFilePath := 'D:\DesktopFiles\Дипломная\NeuralNetForPascal\train_data_enriched.json';
  
  WriteAllText(newFilePath, updatedJson);
end.