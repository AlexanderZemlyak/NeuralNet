uses PABCSystem;
begin
  
  var removedDirs := |'РусскиеИсполнители', 'CheckedTasks', 'Experimental', 'OpenGL и OpenCL'|;
  
  var files := (new System.IO.DirectoryInfo('C:\PABCWork.NET\Samples'))
  .GetFiles('*.pas', System.IO.SearchOption.AllDirectories);
  
  foreach var f in files do
  begin
    
    if removedDirs.Any(d -> f.Directory.FullName.IndexOf(d) <> -1) then
    begin
      // Println(f.Directory);
      continue;
    end;
    
    var destDir := new System.IO.DirectoryInfo('C:\PABCWork.NET\SamplesFlatten\');
    
    var name := f.Name;
    
    var dup := destDir.GetFiles().Find(fileInfo -> fileInfo.Name.ToLower() = name.ToLower());
    
    while dup <> nil do
    begin
      
      var dupName := System.IO.Path.GetFileNameWithoutExtension(dup.Name);
      
      if char.IsDigit(dupName.Last()) then
        name := dupName.Substring(0, dupName.Length - 1) + (integer.Parse(dupName.Last()) + 1).ToString() + '.pas'
      else
        name := dupName + '2' + '.pas';
      
      dup := destDir.GetFiles().Find(fileInfo -> fileInfo.Name.ToLower() = name.ToLower());
    end;
    
    f.CopyTo(destDir.FullName + name);
  end;
  
end.