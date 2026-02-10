{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\Compiler.dll'}
{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\Errors.dll'}
{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\TreeConverter.dll'}
{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\SyntaxTree.dll'}
{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\SemanticTree.dll'}
{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\SyntaxVisitors.dll'}
{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\NetGenerator.dll'}
{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\ParserTools.dll'}
{$reference 'C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\LanguageIntegrator.dll'}
//{$reference CompilerTools.dll}
//{$reference Localization.dll}
//{$reference System.Windows.Forms.dll}
//{$reference LanguageIntegrator.dll}

uses PascalABCCompiler;

const filesCount = 60;

begin
  
  Languages.Integration.LanguageIntegrator.LoadAllLanguages();
  
  for var i := 0 to filesCount - 1 do
  begin

    var baseDir := 'C:\PABCWork.NET\SamplesForNet\';
    
    var fileName := $'{i}.pas';
    
    var fullFilename := baseDir + fileName;
    
    var comp := new Compiler();
    
    var co: CompilerOptions := new CompilerOptions(fullFilename, CompilerOptions.OutputType.ConsoleApplicaton);
    co.Debug := false;
    co.SearchDirectories := ['C:\Users\alex\Desktop\Дипломная\PABCNet source\pascalabcnet\bin\Lib'].ToList();
    co.OutputDirectory := baseDir;
    co.UseDllForSystemUnits := false;
    co.RunWithEnvironment := false; 
    comp.ErrorsList.Clear();
    comp.Warnings.Clear();
    
    comp.Compile(co);
    
    if comp.ErrorsList.Count > 0 then
    begin
      var err := comp.ErrorsList.Last();
      
      Println($'Ошибка при компиляции файла {fileName}:{NewLine}{err}{NewLine}');
      //if not err.ToString().Contains('а ожидался идентификатор') then
      //  System.IO.File.Create(baseDir + $'{i}.exe');
    end
    else
    begin
      var res := comp.ParseText(fullFileName, System.IO.File.ReadAllText(fullFilename), comp.ErrorsList, comp.Warnings);
      var stat := new SyntaxVisitors.ABCStatisticsVisitor();
      stat.ProcessNode(res);
      
      var nh, ph : integer;
      stat.CalcHealth(nh, ph);
      
      if nh > 0 then
      begin
        Println('Bad code style in ' + fileName);
        System.IO.File.Delete(baseDir + $'{i}.exe');
      end;
    end;
    
    
  end;
  
end.