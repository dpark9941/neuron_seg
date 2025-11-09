// HTML에서 버튼을 찾아 변수에 담기
const button = document.getElementById("colorBtn");

// 미리 사용할 색 목록 (배열)
const colors = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#A66CFF"];
let index = 0;  // 현재 색 인덱스

// 버튼 클릭할 때마다 실행되는 함수
button.addEventListener("click", function() {
  // 다음 색 선택
  index = (index + 1) % colors.length;  // 색이 끝나면 다시 처음으로

  // 배경색 바꾸기
  document.body.style.backgroundColor = colors[index];

  // 버튼 글자도 바꾸기
  button.innerText = `지금 색: ${colors[index]}`;
});