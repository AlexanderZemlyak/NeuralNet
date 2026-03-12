uses WPF;

begin
  var s := '''
  <Border Margin="30" Background="Aquamarine" Padding = "15">
    <StackPanel>
      <Button Content = "Button1" HorizontalAlignment = "Left"/>
      <Button Content = "Button2" HorizontalAlignment = "Right"/>
      <Button Content = "Button3" VerticalAlignment = "Bottom"/>
    </StackPanel>
  </Border>
  '''; 
  var scene := Border(ParseXaml(s)).AsMainContent;
  
end.