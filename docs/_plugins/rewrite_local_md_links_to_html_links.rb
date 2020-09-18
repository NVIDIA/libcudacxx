Jekyll::Hooks.register :pages, :post_render do |doc|
  doc.output = doc.output.gsub(/(href="\.\/[^.]*)\.md"/, '\1.html"')
end

