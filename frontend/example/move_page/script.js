// HTML에서 버튼 요소를 찾아오기
const button = document.getElementById("goBtn");

// 클릭 이벤트 추가
button.addEventListener("click", function() {
  // 페이지 이동 코드
  window.location.href = "page2.html";
});