$(document).ready(function(){
  function readURL(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      reader.onload = function (e) {
        $('#img-upload').attr('src', e.target.result);
        }
      reader.readAsDataURL(input.files[0]);
      }
    }
  
  $(".custom-file-input").on("change", function() {
  	var fileName = $(this).val().split("\\").pop();
  	$(this).siblings(".custom-file-label").addClass("selected").html(fileName);
    readURL(this);
    $("#Uploaded-image").show();
	});

  $('#Clear-btn').on('click', function(event) {
    $("#Uploaded-image").hide();
    $("#Predictions-text").hide();
	});
  
});
  
