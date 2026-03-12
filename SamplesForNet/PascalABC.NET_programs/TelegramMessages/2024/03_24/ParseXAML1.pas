uses WPF;

begin
  var s := '''
  <StackPanel Margin="30" Background="Aquamarine">
    <TextBox Height="23" TextWrapping="Wrap" Text="TextBox"/>
    <TextBlock/>
  </StackPanel>
  '''; 
  var scene := StackPanel(ParseXaml(s)).AsMainContent;
  var tb := scene.Children[0] as TextBox;
  var tbl := scene.Children[1] as TextBlock;
  
  tb.TextChanged += procedure (o,e) -> (tbl.Text := tb.Text);
end.