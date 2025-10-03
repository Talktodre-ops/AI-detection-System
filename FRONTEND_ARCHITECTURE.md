# Frontend Architecture Plan - AI Detection System

## 🎯 Project Goals
- Modern, professional landing page
- User authentication (signup/login)
- User dashboard with usage analytics
- AI detection interface
- Subscription/credit-based system

---

## 📊 Tech Stack Options

### **Option 1: Full-Stack Python (Recommended for Quick MVP)** ⭐
**Frontend:** Streamlit + Custom HTML/CSS
**Backend:** FastAPI (already have this!)
**Database:** PostgreSQL or SQLite
**Auth:** JWT + Python libraries

**Pros:**
- ✅ Single language (Python) - easier for you
- ✅ FastAPI backend already built
- ✅ Quick to deploy
- ✅ Good for AI/ML projects

**Cons:**
- ❌ Limited UI customization compared to React
- ❌ Less "modern" feel

---

### **Option 2: Modern React Stack** 🚀
**Frontend:** Next.js 14 (React) + Tailwind CSS + shadcn/ui
**Backend:** FastAPI (Python) - API only
**Database:** PostgreSQL + Supabase
**Auth:** Supabase Auth or NextAuth.js

**Pros:**
- ✅ Beautiful, modern UI
- ✅ Best user experience
- ✅ Scalable and professional
- ✅ Built-in auth with Supabase
- ✅ Great for landing pages

**Cons:**
- ❌ Learning curve (JavaScript/TypeScript)
- ❌ More complex setup

---

### **Option 3: Hybrid Approach** ⚡
**Landing Page:** Next.js (for marketing/SEO)
**App Dashboard:** Streamlit (for AI features)
**Backend:** FastAPI
**Database:** Supabase (PostgreSQL)

**Pros:**
- ✅ Best of both worlds
- ✅ Professional landing page
- ✅ Fast development for AI features

**Cons:**
- ❌ Maintain two frontend codebases

---

## 🗄️ Database Design

### **Recommended: PostgreSQL with Supabase**

#### **Why Supabase?**
- Built-in authentication
- Real-time subscriptions
- RESTful API auto-generated
- Free tier (50,000 monthly active users)
- Easy integration with FastAPI

### **Database Schema:**

```sql
-- Users Table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    password_hash VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    email_verified BOOLEAN DEFAULT FALSE,
    subscription_tier VARCHAR(50) DEFAULT 'free', -- free, pro, enterprise
    credits_remaining INTEGER DEFAULT 100,
    total_credits_used INTEGER DEFAULT 0
);

-- Analysis History Table
CREATE TABLE analysis_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    text_sample TEXT,
    text_length INTEGER,
    prediction VARCHAR(50), -- 'AI' or 'Human'
    confidence FLOAT,
    analysis_type VARCHAR(50), -- 'basic', 'mixed_content', 'fact_check'
    created_at TIMESTAMP DEFAULT NOW(),
    credits_used INTEGER DEFAULT 1
);

-- Subscription Plans Table
CREATE TABLE subscription_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    credits_per_month INTEGER NOT NULL,
    max_text_length INTEGER,
    features JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User Sessions Table (for JWT)
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    ip_address VARCHAR(50),
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Feedback Table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    analysis_id UUID REFERENCES analysis_history(id),
    text TEXT,
    predicted_label VARCHAR(50),
    actual_label VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- API Keys Table (for developers)
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    credits_limit INTEGER,
    rate_limit INTEGER, -- requests per minute
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔐 Authentication Flow

### **Recommended: JWT + Supabase**

```
1. User Signs Up
   ↓
2. Email Verification Link Sent
   ↓
3. User Clicks Link → Email Verified
   ↓
4. User Logs In
   ↓
5. Server Returns JWT Token
   ↓
6. Frontend Stores Token (httpOnly cookie)
   ↓
7. Every API Request Includes Token
   ↓
8. Backend Validates Token → Returns Data
```

### **Security Features:**
- ✅ Password hashing (bcrypt)
- ✅ JWT tokens (short expiry)
- ✅ Refresh tokens
- ✅ Rate limiting
- ✅ CORS protection
- ✅ Input validation

---

## 🎨 User Interface Pages

### **1. Landing Page**
**URL:** `/`

**Sections:**
- Hero section with demo
- Features showcase
- Pricing table
- How it works
- Testimonials
- FAQ
- CTA (Call to Action)

**Tech:** Next.js + Tailwind + Framer Motion

---

### **2. Authentication Pages**

#### **Sign Up** (`/signup`)
- Email
- Full Name
- Password
- Confirm Password
- Terms & Conditions checkbox
- OAuth options (Google, GitHub)

#### **Login** (`/login`)
- Email
- Password
- Remember me
- Forgot password link
- OAuth options

#### **Email Verification** (`/verify-email`)
- Token verification
- Redirect to dashboard

#### **Password Reset** (`/reset-password`)
- Email input
- Token verification
- New password form

---

### **3. User Dashboard** (`/dashboard`)

**Sidebar Navigation:**
- 🏠 Home
- 🔍 AI Detector
- 🔄 Mixed Content
- ✅ Fact Checker
- 📊 Analytics
- 💳 Billing
- ⚙️ Settings

**Main Content:**
- Usage statistics (charts)
- Recent analyses
- Credits remaining
- Quick actions

---

### **4. AI Detection Interface** (`/dashboard/detect`)

**Features:**
- Text input area
- File upload (PDF, DOCX, TXT)
- Detection settings
- Results display
- Export results
- Save to history

---

### **5. Analytics Page** (`/dashboard/analytics`)

**Metrics:**
- Total analyses
- AI vs Human breakdown
- Usage over time (chart)
- Most analyzed content type
- Credits usage

---

### **6. Pricing Page** (`/pricing`)

**Tiers:**

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Credits/month | 100 | 1,000 | Unlimited |
| Max text length | 5,000 chars | 50,000 chars | Unlimited |
| Fact-checking | ❌ | ✅ | ✅ |
| API access | ❌ | ✅ | ✅ |
| Priority support | ❌ | ❌ | ✅ |
| Price | $0 | $29/mo | Custom |

---

## 🔄 Credit System Logic

```python
# Credit Costs
CREDIT_COSTS = {
    "basic_detection": 1,           # Per 1,000 characters
    "mixed_content": 2,             # Per analysis
    "fact_check": 3,                # Per claim checked
    "api_request": 1,               # Per API call
}

# Monthly Limits
SUBSCRIPTION_CREDITS = {
    "free": 100,
    "pro": 1000,
    "enterprise": 999999,
}

# Usage Flow
1. User initiates analysis
2. Calculate credits needed
3. Check if user has enough credits
4. Deduct credits
5. Perform analysis
6. Save to history
7. Return results
```

---

## 📱 API Endpoints Needed

### **Authentication**
```
POST   /api/auth/signup
POST   /api/auth/login
POST   /api/auth/logout
POST   /api/auth/refresh
POST   /api/auth/verify-email
POST   /api/auth/forgot-password
POST   /api/auth/reset-password
```

### **User Management**
```
GET    /api/user/profile
PUT    /api/user/profile
GET    /api/user/credits
GET    /api/user/usage
DELETE /api/user/account
```

### **AI Detection (Protected)**
```
POST   /api/detect/text
POST   /api/detect/file
POST   /api/detect/mixed-content
POST   /api/fact-check
```

### **Analytics**
```
GET    /api/analytics/dashboard
GET    /api/analytics/history
GET    /api/analytics/export
```

### **Billing**
```
GET    /api/billing/plans
POST   /api/billing/subscribe
POST   /api/billing/cancel
GET    /api/billing/invoices
```

---

## 🚀 Deployment Strategy

### **Frontend (Next.js)**
- **Platform:** Vercel (free tier)
- **Domain:** Custom domain + SSL
- **CDN:** Automatic with Vercel

### **Backend (FastAPI)**
- **Platform:** Railway or Render (free tier)
- **Database:** Supabase (free tier)
- **File Storage:** Supabase Storage

### **CI/CD**
- GitHub Actions
- Auto-deploy on push to main

---

## 💡 My Recommendation: **Option 2 (React Stack)**

### **Why?**
1. **Professional & Modern** - Better for attracting users
2. **Scalable** - Easy to add features
3. **Supabase** - Handles auth + database + storage
4. **Fast Development** - Pre-built components (shadcn/ui)
5. **SEO Friendly** - Next.js is great for landing pages
6. **Your FastAPI backend** - Already built, just add auth endpoints

### **Tech Stack:**
```
Frontend:  Next.js 14 + TypeScript + Tailwind CSS + shadcn/ui
Backend:   FastAPI (Python) + JWT
Database:  Supabase (PostgreSQL)
Auth:      Supabase Auth
Payments:  Stripe (when ready)
Hosting:   Vercel (frontend) + Railway (backend)
```

---

## ⏱️ Development Timeline

**Phase 1: Foundation (Week 1)**
- ✅ Set up Next.js project
- ✅ Set up Supabase
- ✅ Design database schema
- ✅ Create landing page

**Phase 2: Auth (Week 2)**
- ✅ Implement signup/login
- ✅ Email verification
- ✅ Password reset
- ✅ Protected routes

**Phase 3: Dashboard (Week 3)**
- ✅ User dashboard UI
- ✅ AI detection interface
- ✅ Connect to FastAPI backend
- ✅ Credit system

**Phase 4: Features (Week 4)**
- ✅ Analytics page
- ✅ Usage history
- ✅ Settings page
- ✅ File upload

**Phase 5: Polish (Week 5)**
- ✅ Responsive design
- ✅ Error handling
- ✅ Loading states
- ✅ Testing

**Phase 6: Launch (Week 6)**
- ✅ Deploy to production
- ✅ Set up monitoring
- ✅ Documentation
- ✅ Beta testing

---

## 🎯 Next Steps

**What do you want to do, Dre?**

1. **Go with Option 2 (React/Next.js)** - I'll create the project structure
2. **Try Option 1 (Streamlit)** - Faster but less polished
3. **Hybrid Approach** - Best of both worlds
4. **Discuss more** - Any questions or changes?

Let me know and I'll start building! 🚀
