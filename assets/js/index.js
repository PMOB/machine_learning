$(function(){
    // スムーズスクロール
    $("a.smooth").click(function(){
        ref = $(this).attr("href");
        target = $(ref == "#" || ref == "" ? "html" : ref);
        pos = target.offset().top;
        if (pos > 24) pos -= 24;
        $("body, html").animate({scrollTop: pos}, 420);
        return false;
    });
});
