// API 호출 최적화를 위한 설정
const API_ENDPOINT = "https://nlu35qe7zd.execute-api.ap-northeast-2.amazonaws.com/creanpl/creanpl";
const BATCH_SIZE = 300; // 동시 처리 개수 제한
const OBSERVER_CONFIG = {
childList: true,
subtree: true,
attributes: false,
characterData: false,
};

// 메인 처리 함수
async function processComments() {
const comments = Array.from(document.querySelectorAll("span.u_cbox_contents"))
    .filter((comment) => comment.textContent.trim() !== "[차단된 내용입니다]"); // 이미 검열된 요소 제외

if (!comments.length) return;

  // API 병렬 처리
for (let i = 0; i < comments.length; i += BATCH_SIZE) {
    const batch = comments.slice(i, i + BATCH_SIZE);
    await Promise.all(batch.map(processSingleComment));
}
}

// 단일 댓글 처리
async function processSingleComment(commentNode) {
try {
    const text = commentNode.textContent.trim();
    if (!text) return;

    const response = await fetch(API_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        com: text,
        lst: [1, 1, 1, 1, 1, 1, 1],
    }),
    });

    if (!response.ok) {
    //console.error(`[검열 시스템] API 요청 실패: HTTP ${response.status}`);
    return;
    }

    const data = await response.json();
    const score = data.score;

    if (typeof score !== "number") return;

    // 조건에 따라 검열 수행
    if (score >= 11) {
    commentNode.textContent = "[차단된 내용입니다]";
    console.log(`[검열 시스템] 차단된 내용: "${text}" / 점수: ${score}`);
    }
} catch (error) {
    //console.error("[검열 시스템] API 요청 중 오류 발생:", error);
}
}

// DOM 변경 감지기 설정
const observer = new MutationObserver((mutations) => {
mutations.forEach((mutation) => {
    if (mutation.addedNodes.length) {
    processComments();
    }
});
});

// 확장 프로그램 활성화 여부에 따른 제어
chrome.storage.local.get("isActive", ({ isActive }) => {
if (isActive) {
    observer.observe(document.body, OBSERVER_CONFIG);
    processComments();
}
});

// 실시간 활성화 상태 변경 감지
chrome.storage.onChanged.addListener((changes) => {
if ("isActive" in changes) {
    if (changes.isActive.newValue) {
    observer.observe(document.body, OBSERVER_CONFIG);
    processComments();
    } else {
    observer.disconnect();
    }
}
});
