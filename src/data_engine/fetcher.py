import openreview
import os
import requests
import time
from typing import List, Dict, Optional
from pathlib import Path

class ConferenceDataFetcher:
    """
    从 OpenReview 获取会议论文数据
    支持获取论文元数据、评审意见、PDF 下载等
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        self.client = None
        self.pdf_dir = os.path.join(output_dir, "pdfs")
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.pdf_dir).mkdir(parents=True, exist_ok=True)
    
    def _init_client(self):
        """初始化 OpenReview 客户端"""
        if self.client is None:
            try:
                # 从环境变量获取认证信息（可选）
                username = os.getenv('OPENREVIEW_USERNAME')
                password = os.getenv('OPENREVIEW_PASSWORD')
                
                self.client = openreview.api.OpenReviewClient(
                    baseurl='https://api2.openreview.net',
                    username=username,
                    password=password
                )
                print("✓ OpenReview client initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenReview client: {e}")
                print("Continuing without authentication (limited access)")
                self.client = openreview.api.OpenReviewClient(
                    baseurl='https://api2.openreview.net'
                )
    
    def fetch_papers(self, venue_id: str = "NeurIPS.cc/2024/Conference", 
                     limit: Optional[int] = None,
                     content_fields: List[str] = None) -> List[Dict]:
        """
        获取会议的论文列表
        
        Args:
            venue_id: 会议 ID（例如 "NeurIPS.cc/2024/Conference"）
            limit: 限制返回的论文数量，None 表示获取全部
            content_fields: 需要获取的内容字段列表
        
        Returns:
            论文字典列表，每个字典包含论文的元数据
        """
        print(f"Fetching papers from {venue_id}...")
        self._init_client()
        
        if content_fields is None:
            content_fields = ['title', 'abstract', 'authors', 'keywords', 'pdf']
        
        papers = []
        
        try:
            # 构建 invitation（投稿邀请）
            submission_invitation = f"{venue_id}/-/Submission"
            
            print(f"Searching for submissions with invitation: {submission_invitation}")
            
            # 获取所有投稿
            submissions = self.client.get_all_notes(
                invitation=submission_invitation,
                details='replies'  # 包含评审等回复信息
            )
            
            print(f"Found {len(submissions)} submissions")
            
            # 处理每篇论文
            for idx, submission in enumerate(submissions):
                if limit and idx >= limit:
                    break
                
                paper_data = self._extract_paper_info(submission, content_fields)
                papers.append(paper_data)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(submissions)} papers")
            
            print(f"✓ Successfully fetched {len(papers)} papers")
            
        except Exception as e:
            print(f"Error fetching papers: {e}")
            import traceback
            traceback.print_exc()
        
        return papers
    
    def _extract_paper_info(self, submission, content_fields: List[str]) -> Dict:
        """
        从 OpenReview submission 对象中提取论文信息
        
        Args:
            submission: OpenReview Note 对象
            content_fields: 需要提取的字段列表
        
        Returns:
            包含论文信息的字典
        """
        paper = {
            "id": submission.id,
            "forum": submission.forum,
            "number": submission.number if hasattr(submission, 'number') else None
        }
        
        # 提取内容字段
        content = submission.content
        
        if 'title' in content_fields:
            paper['title'] = content.get('title', {}).get('value', 'Untitled')
        
        if 'abstract' in content_fields:
            paper['abstract'] = content.get('abstract', {}).get('value', '')
        
        if 'authors' in content_fields:
            authors = content.get('authors', {}).get('value', [])
            paper['authors'] = authors if isinstance(authors, list) else []
        
        if 'keywords' in content_fields:
            keywords = content.get('keywords', {}).get('value', [])
            paper['keywords'] = keywords if isinstance(keywords, list) else []
        
        if 'pdf' in content_fields:
            paper['pdf_url'] = content.get('pdf', {}).get('value', '')
        
        # 提取 TL;DR（如果有）
        paper['tldr'] = content.get('TL;DR', {}).get('value', '')
        
        # 提取决策信息（如果有评审）
        paper['decision'] = None
        paper['reviews'] = []
        
        if hasattr(submission, 'details') and submission.details:
            replies = submission.details.get('replies', [])
            for reply in replies:
                invitation = reply.get('invitation', '')
                
                # 检查是否是评审
                if 'Official_Review' in invitation:
                    review_content = reply.get('content', {})
                    review_data = {
                        'rating': review_content.get('rating', {}).get('value', 'N/A'),
                        'confidence': review_content.get('confidence', {}).get('value', 'N/A'),
                        'summary': review_content.get('summary', {}).get('value', ''),
                        'strengths': review_content.get('strengths', {}).get('value', ''),
                        'weaknesses': review_content.get('weaknesses', {}).get('value', '')
                    }
                    paper['reviews'].append(review_data)
                
                # 检查是否是决策
                elif 'Decision' in invitation:
                    decision_content = reply.get('content', {})
                    paper['decision'] = decision_content.get('decision', {}).get('value', 'Unknown')
        
        return paper
    
    def fetch_paper_by_title(self, title: str) -> Optional[Dict]:
        """
        根据标题搜索特定论文
        
        Args:
            title: 论文标题
        
        Returns:
            论文字典，如果未找到则返回 None
        """
        print(f"Searching for paper: {title}")
        self._init_client()
        
        try:
            # 使用 search_notes 搜索
            notes = self.client.search_notes(term=title, limit=5)
            
            if not notes:
                print(f"  No paper found with title: {title}")
                return None
            
            # 找到最匹配的论文
            title_lower = title.lower().strip()
            for note in notes:
                note_title = note.content.get('title', {}).get('value', '').lower()
                if title_lower in note_title or note_title in title_lower:
                    print(f"  ✓ Found matching paper")
                    return self._extract_paper_info(note, ['title', 'abstract', 'authors', 'pdf'])
            
            # 如果没有完全匹配，返回第一个结果
            print(f"  Using best match (partial)")
            return self._extract_paper_info(notes[0], ['title', 'abstract', 'authors', 'pdf'])
            
        except Exception as e:
            print(f"Error searching for paper: {e}")
            return None
    
    def download_pdfs(self, papers: List[Dict], max_downloads: Optional[int] = None):
        """
        下载论文 PDF
        
        Args:
            papers: 论文列表（需包含 pdf_url 字段）
            max_downloads: 最大下载数量限制
        """
        print(f"Downloading PDFs for {len(papers)} papers...")
        
        downloaded = 0
        failed = 0
        
        for idx, paper in enumerate(papers):
            if max_downloads and downloaded >= max_downloads:
                print(f"Reached download limit ({max_downloads})")
                break
            
            pdf_url = paper.get('pdf_url', '')
            if not pdf_url:
                print(f"  [{idx+1}/{len(papers)}] Skipping (no PDF URL): {paper.get('title', 'Unknown')[:50]}")
                continue
            
            # 生成文件名
            paper_id = paper.get('id', f'paper_{idx}')
            filename = f"{paper_id}.pdf"
            filepath = os.path.join(self.pdf_dir, filename)
            
            # 如果文件已存在，跳过
            if os.path.exists(filepath):
                print(f"  [{idx+1}/{len(papers)}] Already exists: {filename}")
                continue
            
            try:
                # 下载 PDF
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                
                # 保存文件
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded += 1
                print(f"  [{idx+1}/{len(papers)}] ✓ Downloaded: {filename}")
                
                # 避免请求过快
                time.sleep(0.5)
                
            except Exception as e:
                failed += 1
                print(f"  [{idx+1}/{len(papers)}] ✗ Failed: {filename} - {e}")
        
        print(f"\n✓ Download complete: {downloaded} succeeded, {failed} failed")
    
    def get_venue_stats(self, venue_id: str = "NeurIPS.cc/2024/Conference") -> Dict:
        """
        获取会议的统计信息
        
        Args:
            venue_id: 会议 ID
        
        Returns:
            包含统计信息的字典
        """
        print(f"Fetching stats for {venue_id}...")
        self._init_client()
        
        stats = {
            "venue_id": venue_id,
            "total_submissions": 0,
            "accepted": 0,
            "rejected": 0,
            "pending": 0
        }
        
        try:
            submission_invitation = f"{venue_id}/-/Submission"
            submissions = self.client.get_all_notes(
                invitation=submission_invitation,
                details='replies'
            )
            
            stats["total_submissions"] = len(submissions)
            
            # 统计决策
            for submission in submissions:
                if hasattr(submission, 'details') and submission.details:
                    replies = submission.details.get('replies', [])
                    for reply in replies:
                        if 'Decision' in reply.get('invitation', ''):
                            decision = reply.get('content', {}).get('decision', {}).get('value', '').lower()
                            if 'accept' in decision:
                                stats["accepted"] += 1
                            elif 'reject' in decision:
                                stats["rejected"] += 1
                            break
                    else:
                        stats["pending"] += 1
            
            print(f"✓ Stats: {stats['total_submissions']} total, "
                  f"{stats['accepted']} accepted, "
                  f"{stats['rejected']} rejected, "
                  f"{stats['pending']} pending")
            
        except Exception as e:
            print(f"Error fetching stats: {e}")
        
        return stats
