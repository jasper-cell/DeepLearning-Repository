// 翻转链表的递归写法
public ListNode ReverseList(ListNode head){
	if(head == null || head.next == null) return head;
	ListNode next = head.next;
	ListNode newHead = ReverseList(next);
	next.next = head;
	head.next = null;
	return newHead;
}


// 翻转链表的三指针写法
public ListNode ReverseList(ListNode head){
	if(head == null || head.next == null) return head;

	ListNode pre = null;
	ListNode cur = head;
	while(cur != null){
		ListNode next = cur.next;
		cur.next = pre;
		pre = cur;
		cur = next;
	}
	return pre;
}

// 冒泡排序
public void bubble_sort(int[] nums){
	boolean hasChange = true;
	for(int i = 0; i < nums.length - 1; i++){
		hasChange = false;
		for(int j = 0; j < nums.length - i -1; j++){
			if(nums[j+1] < nums[j]){
				swap(nums, j, j+1);
				hasChange = true;
			}
		}
	}
}

// 插入排序
public void insertion_sort(int[] nums){
	for(int i = 1, j, cur; i < nums.length; i++){
		cur = nums[i];
		for(int j = i -1; j >= 0 && nums[j] > cur; j--){
			nums[j+1] = nums[j];
		}
		nums[j+1] = cur;
	}
}

// 选择排序
public void selection_sort(int[] nums){
	for(int i = nums.length - 1; i >= 0; i--){
		int max = j;
		for(int j = 0; j <= i; j++){
			if(nums[j] > nums[max]){
				max = j;
			}
		}
		swap(nums, max, i);
	}
}


// 归并排序
public void merge_sort(int[] nums, int lo, int hi){
	if(lo >= hi) return;

	int mid = lo + (hi - lo) / 2;

	merge_sort(nums, lo, mid);
	merge_sort(nums, mid+1, hi);

	merge(nums, lo, mid, hi);
}

private void merge(int[] nums, int lo, int mid, int hi){
	int[] copy = nums.clone();

	int k = lo, i = lo, j = mid + 1;
	while(k <= hi){
		if(i > mid){
			nums[k++] = copy[j++];
		}else if(j > hi){
			nums[k++] = copy[i++];
		}else if(copy[i] > copy[j]){
			nums[k++] = copy[j++];
		}else{
			nums[k++] = copy[i++];
		}
	}
}


// 快速排序
public void quick_sort(int[] nums, int lo, int hi){
	if(lo >= hi) return;

	int p = partition(nums, lo, hi);

	quick_sort(nums, lo, p-1);
	quick_sort(nums, p+1, hi);
}

private int partition(int[] nums, int lo, int hi){
	int i, j;
	for(i = lo, j = lo; j < hi; j++){
		if(nums[j] <= nums[hi]){
			swap(nums, i++, j);
		}
	}
	swap(nums, i, j);
	return i;
}


// 设计LRU缓存结构
public class Solution{
	static class Node{
		int key, value;
		Node prev, next;
		public Node(int key, int value){
			this.key = key;
			this.value = value;
		}
	}

	private Map<Integer, Node> map = new HashMap<>();
	private Node head = new Node(-1, -1);
	private Node tail = new Node(-1, -1);
	private int k;

	public int[] LRU(int[][] operators, int k){
		// 为map的size大小设置对应的限制
		this.k = k;

		// 将头尾的节点相互连接起来
		head.next = tail;
		tail.prev = head;

		// 建立对应的数组用于存储实际应该返回的值
		int len = (int) Arrays.stream(operators).filter(x -> x[0] == 2).count();
		int[] ans = new int[len];

		int cnt = 0;
		for(int i = 0; i < operators.length; i++){
			if(operators[i][0] == 1){
				set(operators[i][1], operators[i][2]);
			}else{
				ans[cnt++] = get(operators[i][1]);
			}
		}
		return ans;
	}

	private void set(int key, int value){
		// 当mao中满值之后便需要对末尾的节点进行删除
		if(map.size() == k){
			int rk = tail.prev.key;
			tail.prev.prev.next = tail;
			tail.prev = tail.prev.prev;
			map.remove(rk);
		}
		// 后面都需要对map中的该节点进行重新的添加操作
		Node temp = new Node(key, value);
		// 将对应的节点及其Key值添加到map中
		map.add(key, temp);
		// 将新创建的Joe点移动到双端链表的头部
		removeToHead(temp);
	}

	private int get(int key){
		// 首先判断map中是否存在对应的key值
		if(!map.containsKey(key)){
			return -1;
		}else{
			// 如果map中含有该对应的key值，则将其取出
			Node temp = map.get(key);
			temp.next.prev = temp.prev;
			temp.prev.next = temp.next;
			removeToHead(temp);
			return temp.value;
		}
	}

	private void removeToHead(Node node){
		node.next = head.next;
		head.next.prev = node;
		head.next = node;
		node.prev = head;
	}
}

// 判断链表中是否有环
public boolean hasCycle(ListNode head){
	if(head == null || head.next == null) return false;
	ListNode fast = head.next.next;
	ListNode slow = head.next;

	while(fast != null && fast.next != null){
		if(fast == slow) return true;
		fast = fast.next.next;
		slow = slow.next;
	}
	return false;
}

// 用两个栈实现队列
public class Solution{
	Stack<Integer> stack1 = new Stack<Integer>();
	Stack<Integer> stack2 = new Stack<Integer>();

	public void push(int node){
		stack1.push(node);
	}

	public int pop(){
		if(stack2.isEmpty()){
			while(!stack1.isEmpty()){
				stack2.push(stack1.pop());
			}
		}
		int re = stack2.pop();
		return re;
	}
}

// 二分查找-II
// 一般都是会在等于的时候进行一些额外的判断
public int search(int[] nums, int target)
{
	if(nums == null || nums.length == 0) return -1;

	int l = 0, r = nums.length - 1;
	while(l <= r){
		int m = l + (r - l) / 2;
		if(nums[m] < target){
			l = m + 1;
		}else if(nums[m] > target){
			r = m -1;
		}else if(nums[m] == target){
			if(m >= 1 && nums[m-1] == target){
				r = m -1;
			}else{
				return m;
			}
		}
	}
	return -1;
}

// 二叉树的层序遍历
public class Solution
{
	public ArraysList<ArraysList<Integer>> leverlorder (TreeNode root){
		if(root == null) return new ArraysList<>();

		ArraysList<ArraysList<Integer>> res = new ArraysList<>();
		Queue<TreeNode> queue = new LinkedList<>();
		queue.add(root);
		while(!queue.isEmpty()){
			int size = queue.size();
			ArraysList<Integer> subList = new ArraysList<>();
			for(int i = 0; i < size; i ++){
				TreeNode temp = queue.poll();
				subList.add(temp.val);
				if(temp.left != null) queue.add(temp.left);
				if(temp.right != null) queue.add(temp.right);
			}
			res.add(subList);
		}
		return res;
	}
}

// 跳台阶问题
public int jumpFloor(int target)
{	
	if(target <= 1) return 1;
	return jumpFloor(target - 1) + jumpFloor(target - 2);
}

public int maxLength(int[] arr)
{
	if(arr.length < 2) return arr.length;

	HashMap<Integer, Integer> map = new HashMap<>();
	int res = 0;
	int left = -1;

	for(int right = 0; right < arr.length; right++)
	{
		if(map.containsKey(arr[right])){
			left = Math.max(left, map.get(arr[right]));
		}
		res = Math.max(res, right - left);
		map.put(arr[right], right);
	}
	return res;
}

// 找树中两个节点的公共祖先
public int lowestCommonAncestor(TreeNode root, int o1, int o2)
{
	return helper(root, o1, o2).val;
}

public TreeNode helper(TreeNode root, int o1, int o2){
	if(root == null || root.val == o1 || root.val == o2){
		return root;
	}

	TreeNode left = helper(root.left, o1, o2);
	TreeNode right = helper(root.right, o1, o2);
	return left == ? right : right == null ? left : root;
}

// 链表是否有环及其入口
public class Solution {

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if(pHead == null || pHead.next == null) return null; // 有一个节点的时候返回null
        ListNode fast = pHead;
        ListNode slow = pHead;
        while(fast != null && fast.next != null){
            // 先写往前走的步骤，防止第一次就直接跳出
            fast = fast.next.next;
            slow = slow.next;
            // 两者相遇的时候，再把其中一个指针移动到头部继续同步往前走
            if(fast == slow){
                fast = pHead;
                while(fast != slow){
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }
}

// 合并两个排序链表
public class Solution {
    public ListNode Merge(ListNode list1,ListNode list2) {
        if(list1 == null) return list2;
        if(list2 == null) return list1;
        
        if(list1.val > list2.val)
        {
            list2.next = Merge(list1, list2.next);
            return list2;
        }else{
            list1.next = Merge(list1.next, list2);
            return list1;
        }
    }
}

// 单链表排序
public class Solution {
    /**
     * 
     * @param head ListNode类 the head node
     * @return ListNode类
     */
    public ListNode sortInList (ListNode head) {
        // write code here
        if (head == null || head.next == null)
            return head;
        // 使用快慢指针寻找链表的中点
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        // 递归左右两边进行排序
        ListNode left = sortInList(head);
        ListNode right = sortInList(tmp);
        // 创建新的链表
        ListNode h = new ListNode(0);
        ListNode res = h;
        // 合并 left right两个链表
        while (left != null && right != null) {
            // left  right链表循环对比
            if (left.val < right.val) {
                h.next = left;
                left = left.next;
            } else {
                h.next = right;
                right = right.next;
            }
            h = h.next;
        }
        // 最后添加未对比的链表部分判断左链表是否为空
        h.next = left != null ? left : right;
        return res.next;
    }
}


// 最小的k个数
import java.util.ArrayList;
import java.util.*;

public class Solution {
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        Arrays.sort(input);
        ArrayList<Integer> res = new ArrayList<>();
        for(int i = 0; i < k; i++){
            res.add(input[i]);
        }
        return res;
    }
}

// 合并两个有序数组
public class Solution {
    public void merge(int A[], int m, int B[], int n) {
        int indexMerge = m + n -1;
        int indexA = m - 1;
        int indexB = n - 1;
        while(indexB >= 0){
            if(indexA < 0){
                A[indexMerge--] = B[indexB--];
            }else if(A[indexA] < B[indexB]){
                A[indexMerge--] = B[indexB--];
            }else{
                A[indexMerge--] = A[indexA--];
            }
        }
    }
}


// 树的三种遍历方式
import java.util.*;

/*
 * public class TreeNode {
 *   int val = 0;
 *   TreeNode left = null;
 *   TreeNode right = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param root TreeNode类 the root of binary tree
     * @return int整型二维数组
     */
    
    public int[][] threeOrders (TreeNode root) {
        // write code here
        ArrayList<Integer> pre = new ArrayList<>();
        ArrayList<Integer> mid = new ArrayList<>();
        ArrayList<Integer> post = new ArrayList<>();
        
        preOrder(root, pre);
        inOrder(root, mid);
        postOrder(root, post);
        
        int[][] res = new int[3][post.size()];
        for(int i = 0; i< post.size(); i++){
            res[0][i] = pre.get(i);
            res[1][i] = mid.get(i);
            res[2][i] = post.get(i);
        }
        return res;
    }
    
    private void preOrder(TreeNode root, ArrayList pre){
        if(root == null){
            return;
        }
        pre.add(root.val);
        preOrder(root.left, pre);
        preOrder(root.right, pre);
    }
    
    private void inOrder(TreeNode root, ArrayList mid){
        if(root == null) return;
        
        inOrder(root.left, mid);
        mid.add(root.val);
        inOrder(root.right, mid);
    }
    
    private void postOrder(TreeNode root, ArrayList post){
        if(root == null)return;
        
        postOrder(root.left, post);
        postOrder(root.right, post);
        post.add(root.val);
    }
    
}

// 寻找第k大
import java.util.*;

public class Solution {
    public int findKth(int[] a, int n, int K) {
        // write code here
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for(int val : a){
            pq.add(val);
            if(pq.size() > K){
                pq.poll();
            }
        }
        return pq.peek();
    }
}


// 子数组最大累加和
import java.util.*;


public class Solution {
    /**
     * max sum of the subarray
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxsumofSubarray (int[] arr) {
        // write code here
        // 存放临时答案
        int thisSum = 0;
        // 存放最终答案，注意初始化的值
        int ans = Integer.MIN_VALUE;
        int len = arr.length;
        for(int i = 0; i < len; i++){
            // 每次先对临时答案进行求和
            thisSum += arr[i];
            
            // 如果临时答案大于要求的值，则更新最终答案
            if(thisSum > ans){
                ans = thisSum;
            }
            
            // 如果临时答案变为负数，则将此临时答案置为0
            if(thisSum < 0){
                thisSum = 0;
            }
        }
        return ans;
    }
}


// 反转字符串
import java.util.*;


public class Solution {
    /**
     * 反转字符串
     * @param str string字符串 
     * @return string字符串
     */
    public String solve (String str) {
        // write code here
        int left = 0, right = str.length() - 1;
        char[] res = str.toCharArray();
        while(left < right){
            char cl = str.charAt(left);
            char cr = str.charAt(right);
            res[left++] = cr;
            res[right--] = cl;
        }
        return new String(res);
    }
}



